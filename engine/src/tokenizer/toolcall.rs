//! Incremental reasoning/content/tool framing, with explicitly selected checkpoint grammars.
//!
//! The default remains Qwen. [`ToolGrammar::Flint`] uses
//! `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>`.
//! Only the call-body grammar differs; both share streaming framing and schema typing.
//!
//! The pinned chat templates render an assistant tool call as
//! `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`
//! (`engine/testdata/chat_template.jinja`). A call without arguments has no parameter block.
//! Reasoning sits between `<think>` and `</think>`. The generation prompt already ends with
//! `<think>\n` when thinking is on, so the generated text then starts inside the reasoning
//! ([`Start::Reasoning`]).
//!
//! The parser works on decoded text, not on token ids. `<think>`, `</think>`, `<tool_call>`, and
//! `</tool_call>` are ordinary added tokens (ids 248068, 248069, 248058, 248059) and decode to
//! their text.
//!
//! One state machine serves streaming and whole-text parsing. [`ToolCallParser::push`] takes the
//! decoded pieces in order. It never emits a partial marker as text: it holds back the longest
//! tail of the buffer that could still grow into a marker. [`parse_complete`] feeds one piece and
//! finishes, so both paths share one rule set.
//!
//! Rules:
//! - Whitespace at the start and end of each reasoning or content section is dropped, as the
//!   template does with `|trim` when it renders the same text back.
//! - `Start::Reasoning` knows only `</think>`. `Start::Content` knows `<think>` and `<tool_call>`.
//!   A marker of the other mode is plain text.
//! - A parameter value is the text between `<parameter=KEY>\n` and `\n</parameter>`. The first
//!   `</parameter>` ends it. A value that the tool schema declares as `"type": "string"` stays a
//!   string. Any other value is parsed as JSON and falls back to a string when it is not JSON.
//! - A body that does not follow the format becomes [`ToolEvent::Malformed`], never content. The
//!   caller decides what to do with it.

use super::ToolSpec;
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

#[cfg(test)]
#[path = "toolcall_flint_tests.rs"]
mod flint_tests;

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

/// The tool syntax selected by the checkpoint's pinned template, never by output sniffing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolGrammar {
    #[default]
    Qwen,
    Flint,
}

impl ToolGrammar {
    pub(super) fn for_template(source: &str) -> Self {
        use sha2::{Digest, Sha256};
        // RecursiveSI/flint 2a247be0 and flint-int4 3b93f905 share these exact bytes.
        let hash = format!("{:x}", Sha256::digest(source.as_bytes()));
        if hash == "7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466" {
            Self::Flint
        } else {
            Self::Qwen
        }
    }
}

/// Where the generated text starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Start {
    /// The text starts as content. The prompt ended with an empty think block or with no think block.
    Content,
    /// The text starts inside the reasoning. The prompt ended with `<think>\n`.
    Reasoning,
}

/// The parameters that the request tools declare as strings.
#[derive(Debug, Clone, Default)]
pub struct ToolSchemas {
    string_params: HashMap<String, HashSet<String>>,
}

impl ToolSchemas {
    /// Reads the string parameters from the request tools.
    ///
    /// A tool is the OpenAI `{"type": "function", "function": {...}}` object or the bare
    /// `{"name": ..., "parameters": ...}` object. A tool without a name is skipped.
    pub fn from_tools(tools: &[ToolSpec]) -> Self {
        let mut string_params: HashMap<String, HashSet<String>> = HashMap::new();
        for tool in tools {
            let definition = tool.0.get("function").unwrap_or(&tool.0);
            let Some(name) = definition.get("name").and_then(Value::as_str) else {
                continue;
            };
            let entry = string_params.entry(name.to_string()).or_default();
            let properties = definition
                .get("parameters")
                .and_then(|parameters| parameters.get("properties"))
                .and_then(Value::as_object);
            for (key, schema) in properties.into_iter().flatten() {
                if schema.get("type").and_then(Value::as_str) == Some("string") {
                    entry.insert(key.clone());
                }
            }
        }
        Self { string_params }
    }

    /// Returns true when `tool` declares `parameter` as a string.
    fn is_string(&self, tool: &str, parameter: &str) -> bool {
        self.string_params
            .get(tool)
            .is_some_and(|params| params.contains(parameter))
    }
}

/// One complete tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// The call id: the parser prefix followed by the position of the call in this response.
    pub id: String,
    /// The function name.
    pub name: String,
    /// The arguments as compact JSON object text, in the order the model wrote them.
    pub arguments: String,
}

/// Why a tool-call body is not a valid call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallError {
    /// The body does not start with `<function=`.
    MissingFunction,
    /// A function or parameter name is empty, holds whitespace, or holds `<`.
    InvalidName,
    /// A `<function=` or `<parameter=` tag has no closing `>`.
    UnclosedTag,
    /// A parameter has no `</parameter>` (Qwen) or `</arg_value>` (Flint).
    UnclosedParameter,
    /// The body ends before `</function>`.
    MissingFunctionClose,
    /// Text that is not a parameter block stands between or after the blocks.
    UnexpectedText,
    /// Two parameters have this name.
    DuplicateParameter(String),
    /// The text ended before `</tool_call>`.
    Unterminated,
}

/// A tool-call body that is not valid, with the text between the call tags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MalformedCall {
    /// The reason.
    pub error: ToolCallError,
    /// The text between `<tool_call>` and `</tool_call>`, or after `<tool_call>` when unterminated.
    pub raw: String,
}

/// One result of the parser, in output order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolEvent {
    /// Reasoning text.
    Reasoning(String),
    /// Content text.
    Content(String),
    /// A complete valid tool call.
    Call(ToolCall),
    /// A tool-call body that is not valid.
    Malformed(MalformedCall),
}

/// The streaming parser. One parser serves one response.
#[derive(Debug)]
pub struct ToolCallParser {
    grammar: ToolGrammar,
    schemas: ToolSchemas,
    id_prefix: String,
    calls: usize,
    reasoning: bool,
    in_call: bool,
    buffer: String,
    leading: bool,
    held: String,
}

impl ToolCallParser {
    /// Creates a parser. Call ids are `id_prefix` followed by 0, 1, and so on.
    pub fn new(start: Start, schemas: ToolSchemas, id_prefix: &str) -> Self {
        Self::with_grammar(start, schemas, id_prefix, ToolGrammar::Qwen)
    }

    /// Creates a parser for the checkpoint's grammar. Never infer syntax from output:
    /// a malformed Flint call must not become a valid Qwen call, or vice versa.
    pub fn with_grammar(
        start: Start,
        schemas: ToolSchemas,
        id_prefix: &str,
        grammar: ToolGrammar,
    ) -> Self {
        Self {
            grammar,
            schemas,
            id_prefix: id_prefix.to_string(),
            calls: 0,
            reasoning: start == Start::Reasoning,
            in_call: false,
            buffer: String::new(),
            leading: true,
            held: String::new(),
        }
    }

    /// Feeds the next decoded piece and returns what became final.
    pub fn push(&mut self, piece: &str) -> Vec<ToolEvent> {
        self.buffer.push_str(piece);
        let mut events = Vec::new();
        self.drain(&mut events);
        events
    }

    /// Ends the text. A held partial marker becomes text. An open call becomes `Unterminated`.
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        let mut events = Vec::new();
        self.drain(&mut events);
        let rest = std::mem::take(&mut self.buffer);
        if self.in_call {
            self.in_call = false;
            events.push(ToolEvent::Malformed(MalformedCall {
                error: ToolCallError::Unterminated,
                raw: rest,
            }));
        } else {
            self.text(&mut events, &rest);
        }
        self.held.clear();
        events
    }

    /// Consumes every marker and every safe text prefix in the buffer.
    fn drain(&mut self, events: &mut Vec<ToolEvent>) {
        loop {
            if self.in_call {
                let Some(position) = self.buffer.find(CALL_CLOSE) else {
                    return;
                };
                let body = take_front(&mut self.buffer, position);
                take_front(&mut self.buffer, CALL_CLOSE.len());
                self.in_call = false;
                self.leading = true;
                events.push(self.call_event(body));
                continue;
            }
            let markers: &[&str] = if self.reasoning {
                &[THINK_CLOSE]
            } else {
                &[THINK_OPEN, CALL_OPEN]
            };
            let hit = markers
                .iter()
                .filter_map(|marker| self.buffer.find(marker).map(|at| (at, *marker)))
                .min_by_key(|(at, _)| *at);
            let Some((position, marker)) = hit else {
                let keep = partial_marker_len(&self.buffer, markers);
                let cut = self.buffer.len() - keep;
                let safe = take_front(&mut self.buffer, cut);
                self.text(events, &safe);
                return;
            };
            let before = take_front(&mut self.buffer, position);
            take_front(&mut self.buffer, marker.len());
            self.text(events, &before);
            self.held.clear();
            self.leading = true;
            match marker {
                THINK_CLOSE => self.reasoning = false,
                THINK_OPEN => self.reasoning = true,
                _ => self.in_call = true,
            }
        }
    }

    /// Emits section text with the section trim rules.
    fn text(&mut self, events: &mut Vec<ToolEvent>, text: &str) {
        let mut text = text;
        if self.leading {
            text = text.trim_start();
            if text.is_empty() {
                return;
            }
            self.leading = false;
        } else if text.is_empty() {
            return;
        }
        let body = text.trim_end();
        if body.is_empty() {
            self.held.push_str(text);
            return;
        }
        let mut piece = std::mem::take(&mut self.held);
        piece.push_str(body);
        self.held = text.get(body.len()..).unwrap_or_default().to_string();
        events.push(if self.reasoning {
            ToolEvent::Reasoning(piece)
        } else {
            ToolEvent::Content(piece)
        });
    }

    /// Turns one call body into a call or a malformed event.
    fn call_event(&mut self, body: String) -> ToolEvent {
        let parsed = match self.grammar {
            ToolGrammar::Qwen => parse_body(&body, &self.schemas),
            ToolGrammar::Flint => parse_flint_body(&body, &self.schemas),
        };
        match parsed {
            Ok((name, arguments)) => {
                let id = format!("{}{}", self.id_prefix, self.calls);
                self.calls += 1;
                ToolEvent::Call(ToolCall {
                    id,
                    name,
                    arguments: Value::Object(arguments).to_string(),
                })
            }
            Err(error) => ToolEvent::Malformed(MalformedCall { error, raw: body }),
        }
    }
}

/// Removes and returns the first `count` bytes. `count` must end on a character boundary.
fn take_front(buffer: &mut String, count: usize) -> String {
    let rest = buffer.split_off(count);
    std::mem::replace(buffer, rest)
}

/// Returns the length of the longest buffer tail that is a proper prefix of a marker.
fn partial_marker_len(buffer: &str, markers: &[&str]) -> usize {
    let longest = markers.iter().map(|marker| marker.len()).max().unwrap_or(1) - 1;
    (1..=longest.min(buffer.len()))
        .rev()
        .find(|&count| {
            buffer.get(buffer.len() - count..).is_some_and(|tail| {
                markers
                    .iter()
                    .any(|marker| marker.len() > count && marker.starts_with(tail))
            })
        })
        .unwrap_or(0)
}

/// Checks a function or parameter name.
fn check_name(name: &str) -> Result<(), ToolCallError> {
    if name.is_empty() || name.contains(char::is_whitespace) || name.contains('<') {
        return Err(ToolCallError::InvalidName);
    }
    Ok(())
}

/// Parses one call body into the function name and the typed arguments.
fn parse_body(
    body: &str,
    schemas: &ToolSchemas,
) -> Result<(String, Map<String, Value>), ToolCallError> {
    let rest = body
        .trim_start()
        .strip_prefix(FUNCTION_OPEN)
        .ok_or(ToolCallError::MissingFunction)?;
    let (name, mut rest) = rest.split_once('>').ok_or(ToolCallError::UnclosedTag)?;
    check_name(name)?;
    let mut arguments = Map::new();
    loop {
        rest = rest.trim_start();
        if let Some(after) = rest.strip_prefix(FUNCTION_CLOSE) {
            if after.trim().is_empty() {
                return Ok((name.to_string(), arguments));
            }
            return Err(ToolCallError::UnexpectedText);
        }
        let Some(after) = rest.strip_prefix(PARAMETER_OPEN) else {
            return Err(if rest.is_empty() {
                ToolCallError::MissingFunctionClose
            } else {
                ToolCallError::UnexpectedText
            });
        };
        let (key, after) = after.split_once('>').ok_or(ToolCallError::UnclosedTag)?;
        check_name(key)?;
        let (value, after) = after
            .split_once(PARAMETER_CLOSE)
            .ok_or(ToolCallError::UnclosedParameter)?;
        if arguments.contains_key(key) {
            return Err(ToolCallError::DuplicateParameter(key.to_string()));
        }
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        arguments.insert(key.to_string(), argument_value(schemas, name, key, value));
        rest = after;
    }
}

fn argument_value(schemas: &ToolSchemas, name: &str, key: &str, value: &str) -> Value {
    if schemas.is_string(name, key) {
        Value::String(value.to_string())
    } else {
        serde_json::from_str(value.trim()).unwrap_or_else(|_| Value::String(value.to_string()))
    }
}

/// Parses only the Flint body. Argument values preserve their bytes (unlike Qwen,
/// Flint has no newline separators to remove). The first closing value marker wins.
fn parse_flint_body(
    body: &str,
    schemas: &ToolSchemas,
) -> Result<(String, Map<String, Value>), ToolCallError> {
    let (name, mut rest) = body.split_once("<arg_key>").unwrap_or((body, ""));
    let name = name.trim();
    check_name(name)?;
    let mut arguments = Map::new();
    if rest.is_empty() && !body.contains("<arg_key>") {
        return Ok((name.to_string(), arguments));
    }
    loop {
        let (key, after) = rest
            .split_once("</arg_key>")
            .ok_or(ToolCallError::UnclosedTag)?;
        check_name(key)?;
        let after = after
            .trim_start()
            .strip_prefix("<arg_value>")
            .ok_or(ToolCallError::UnexpectedText)?;
        let (value, after) = after
            .split_once("</arg_value>")
            .ok_or(ToolCallError::UnclosedParameter)?;
        if arguments.contains_key(key) {
            return Err(ToolCallError::DuplicateParameter(key.to_string()));
        }
        arguments.insert(key.to_string(), argument_value(schemas, name, key, value));
        let after = after.trim();
        if after.is_empty() {
            return Ok((name.to_string(), arguments));
        }
        rest = after
            .strip_prefix("<arg_key>")
            .ok_or(ToolCallError::UnexpectedText)?;
    }
}

/// The result of a whole-text parse.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedOutput {
    /// The reasoning text.
    pub reasoning: String,
    /// The content text.
    pub content: String,
    /// The complete valid tool calls, in order.
    pub tool_calls: Vec<ToolCall>,
    /// The invalid call bodies, in order.
    pub malformed: Vec<MalformedCall>,
}

impl ParsedOutput {
    /// Folds parser events into one result. Text events of one kind are joined.
    pub fn from_events(events: impl IntoIterator<Item = ToolEvent>) -> Self {
        let mut output = Self::default();
        for event in events {
            match event {
                ToolEvent::Reasoning(text) => output.reasoning.push_str(&text),
                ToolEvent::Content(text) => output.content.push_str(&text),
                ToolEvent::Call(call) => output.tool_calls.push(call),
                ToolEvent::Malformed(malformed) => output.malformed.push(malformed),
            }
        }
        output
    }
}

/// Parses a complete response text. This is the non-streaming path.
pub fn parse_complete(
    text: &str,
    start: Start,
    schemas: ToolSchemas,
    id_prefix: &str,
) -> ParsedOutput {
    let mut parser = ToolCallParser::new(start, schemas, id_prefix);
    let mut events = parser.push(text);
    events.extend(parser.finish());
    ParsedOutput::from_events(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{ChatTemplate, PromptMessage, Role, Thinking, ToolCallSpec};
    use serde_json::json;

    /// Three tools: `write_file` and `run_python` take strings, `add` takes typed values.
    fn tools() -> Vec<ToolSpec> {
        vec![
            ToolSpec(json!({"type": "function", "function": {
                "name": "write_file",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"}, "content": {"type": "string"}}}}})),
            ToolSpec(json!({"type": "function", "function": {
                "name": "run_python",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}})),
            ToolSpec(json!({"type": "function", "function": {
                "name": "add",
                "parameters": {"type": "object", "properties": {
                    "a": {"type": "integer"}, "b": {"type": "integer"},
                    "flag": {"type": "boolean"}, "tags": {"type": "array"},
                    "label": {"type": "string"}}}}})),
        ]
    }

    fn parse(text: &str, start: Start) -> ParsedOutput {
        parse_complete(text, start, ToolSchemas::from_tools(&tools()), "call_")
    }

    fn parser(start: Start) -> ToolCallParser {
        ToolCallParser::new(start, ToolSchemas::from_tools(&tools()), "call_")
    }

    fn call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: arguments.to_string(),
        }
    }

    fn wrap(body: &str) -> String {
        format!("<tool_call>{body}</tool_call>")
    }

    /// Asserts that `body` inside call tags is reported as `error` and nowhere else.
    fn assert_malformed(body: &str, error: ToolCallError) {
        let output = parse(&wrap(body), Start::Content);
        assert_eq!(output.content, "", "malformed body must not become content");
        assert!(
            output.tool_calls.is_empty(),
            "calls: {:?}",
            output.tool_calls
        );
        assert_eq!(
            output.malformed,
            [MalformedCall {
                error,
                raw: body.to_string()
            }]
        );
    }

    #[test]
    fn plain_text_is_content() {
        let output = parse("Hello world", Start::Content);
        let expected = ParsedOutput {
            content: "Hello world".to_string(),
            ..ParsedOutput::default()
        };
        assert_eq!(output, expected);
    }

    #[test]
    fn reasoning_start_splits_at_the_close_marker() {
        let output = parse(
            "I should check.\n</think>\n\nThe answer is 5.",
            Start::Reasoning,
        );
        assert_eq!(output.reasoning, "I should check.", "reasoning");
        assert_eq!(output.content, "The answer is 5.", "content");
    }

    #[test]
    fn content_start_reads_an_inline_think_block() {
        let output = parse("<think>\nplan\n</think>\n\nDone", Start::Content);
        assert_eq!(output.reasoning, "plan", "reasoning");
        assert_eq!(output.content, "Done", "content");
    }

    #[test]
    fn whitespace_at_both_ends_of_a_section_is_dropped() {
        let output = parse("  \n lead and trail \n\n", Start::Content);
        assert_eq!(output.content, "lead and trail");
    }

    #[test]
    fn markers_of_the_other_mode_stay_text() {
        let output = parse("a <tool_call> b</think>c </think> d", Start::Reasoning);
        assert_eq!(output.reasoning, "a <tool_call> b", "reasoning");
        assert_eq!(output.content, "c </think> d", "content");
    }

    #[test]
    fn one_call_types_its_arguments_by_schema() {
        let text = concat!(
            "I will add them.\n\n<tool_call>\n<function=add>\n",
            "<parameter=a>\n2\n</parameter>\n",
            "<parameter=b>\n3\n</parameter>\n",
            "<parameter=flag>\ntrue\n</parameter>\n",
            "<parameter=tags>\n[\"x\", \"y\"]\n</parameter>\n",
            "<parameter=label>\n7\n</parameter>\n",
            "</function>\n</tool_call>"
        );
        let output = parse(text, Start::Content);
        assert_eq!(output.content, "I will add them.", "content");
        assert_eq!(
            output.tool_calls,
            [call(
                "call_0",
                "add",
                r#"{"a":2,"b":3,"flag":true,"tags":["x","y"],"label":"7"}"#
            )]
        );
        assert!(
            output.malformed.is_empty(),
            "malformed: {:?}",
            output.malformed
        );
    }

    #[test]
    fn a_multi_line_string_value_keeps_its_inner_newlines() {
        let text = concat!(
            "<tool_call>\n<function=write_file>\n",
            "<parameter=path>\nhello.py\n</parameter>\n",
            "<parameter=content>\nline1\nline2\n\nline4\n</parameter>\n",
            "</function>\n</tool_call>"
        );
        let output = parse(text, Start::Content);
        assert_eq!(
            output.tool_calls,
            [call(
                "call_0",
                "write_file",
                r#"{"path":"hello.py","content":"line1\nline2\n\nline4"}"#
            )]
        );
    }

    #[test]
    fn a_string_parameter_that_looks_like_json_stays_a_string() {
        let text = concat!(
            "<tool_call>\n<function=write_file>\n",
            "<parameter=content>\n{\"a\": 1}\n</parameter>\n",
            "</function>\n</tool_call>"
        );
        let output = parse(text, Start::Content);
        assert_eq!(
            output.tool_calls,
            [call("call_0", "write_file", r#"{"content":"{\"a\": 1}"}"#)]
        );
    }

    #[test]
    fn an_unknown_tool_parses_json_values_and_keeps_other_text() {
        let text = concat!(
            "<tool_call>\n<function=mystery>\n",
            "<parameter=n>\n12\n</parameter>\n",
            "<parameter=s>\nhello world\n</parameter>\n",
            "<parameter=t>\n[1, 2\n</parameter>\n",
            "<parameter=z>\nnull\n</parameter>\n",
            "</function>\n</tool_call>"
        );
        let output = parse(text, Start::Content);
        assert_eq!(
            output.tool_calls,
            [call(
                "call_0",
                "mystery",
                r#"{"n":12,"s":"hello world","t":"[1, 2","z":null}"#
            )]
        );
    }

    #[test]
    fn a_bare_tool_definition_without_the_function_wrapper_declares_strings() {
        let bare = ToolSpec(json!({
            "name": "echo",
            "parameters": {"properties": {"text": {"type": "string"}}}
        }));
        let text = "<tool_call>\n<function=echo>\n<parameter=text>\n42\n</parameter>\n</function>\n</tool_call>";
        let output = parse_complete(text, Start::Content, ToolSchemas::from_tools(&[bare]), "c");
        assert_eq!(output.tool_calls, [call("c0", "echo", r#"{"text":"42"}"#)]);
    }

    #[test]
    fn two_calls_keep_their_order_and_get_distinct_ids() {
        let text = concat!(
            "<tool_call>\n<function=write_file>\n",
            "<parameter=path>\nhello.py\n</parameter>\n",
            "<parameter=content>\nprint(2 + 3)\n</parameter>\n",
            "</function>\n</tool_call>\n",
            "<tool_call>\n<function=run_python>\n",
            "<parameter=path>\nhello.py\n</parameter>\n",
            "</function>\n</tool_call>"
        );
        let output = parse(text, Start::Content);
        assert_eq!(output.content, "", "content between calls");
        assert_eq!(
            output.tool_calls,
            [
                call(
                    "call_0",
                    "write_file",
                    r#"{"path":"hello.py","content":"print(2 + 3)"}"#
                ),
                call("call_1", "run_python", r#"{"path":"hello.py"}"#),
            ]
        );
    }

    #[test]
    fn reasoning_then_a_call() {
        let text = concat!(
            "Need a file.\n</think>\n\n",
            "<tool_call>\n<function=run_python>\n",
            "<parameter=path>\nhello.py\n</parameter>\n",
            "</function>\n</tool_call>"
        );
        let output = parse(text, Start::Reasoning);
        assert_eq!(output.reasoning, "Need a file.", "reasoning");
        assert_eq!(output.content, "", "content");
        assert_eq!(
            output.tool_calls,
            [call("call_0", "run_python", r#"{"path":"hello.py"}"#)]
        );
    }

    #[test]
    fn a_call_without_arguments_has_an_empty_object() {
        let output = parse(
            "<tool_call>\n<function=ping>\n</function>\n</tool_call>",
            Start::Content,
        );
        assert_eq!(output.tool_calls, [call("call_0", "ping", "{}")]);
    }

    #[test]
    fn call_ids_start_with_the_given_prefix() {
        let text = "<tool_call>\n<function=ping>\n</function>\n</tool_call>";
        let output = parse_complete(text, Start::Content, ToolSchemas::default(), "chatcmpl-9_");
        assert_eq!(output.tool_calls[0].id, "chatcmpl-9_0");
    }

    #[test]
    fn call_ids_count_only_valid_calls() {
        let text = concat!(
            "<tool_call>\nbad\n</tool_call>",
            "<tool_call>\n<function=ping>\n</function>\n</tool_call>"
        );
        let output = parse(text, Start::Content);
        assert_eq!(output.malformed.len(), 1, "malformed");
        assert_eq!(output.tool_calls, [call("call_0", "ping", "{}")]);
    }

    #[test]
    fn a_json_style_body_is_missing_the_function_tag() {
        assert_malformed(
            "\n{\"name\": \"add\", \"arguments\": {\"a\": 1}}\n",
            ToolCallError::MissingFunction,
        );
    }

    #[test]
    fn an_empty_function_name_is_invalid() {
        assert_malformed("\n<function=>\n</function>\n", ToolCallError::InvalidName);
    }

    #[test]
    fn a_function_name_with_whitespace_is_invalid() {
        assert_malformed(
            "\n<function=add now>\n</function>\n",
            ToolCallError::InvalidName,
        );
    }

    #[test]
    fn a_function_tag_without_a_closing_bracket_is_unclosed() {
        assert_malformed("\n<function=add", ToolCallError::UnclosedTag);
    }

    #[test]
    fn a_parameter_without_a_close_tag_is_unclosed() {
        assert_malformed(
            "\n<function=add>\n<parameter=a>\n1\n</function>\n",
            ToolCallError::UnclosedParameter,
        );
    }

    #[test]
    fn a_body_that_ends_before_the_function_close_is_reported() {
        assert_malformed(
            "\n<function=add>\n<parameter=a>\n1\n</parameter>\n",
            ToolCallError::MissingFunctionClose,
        );
    }

    #[test]
    fn stray_text_between_parameter_blocks_is_reported() {
        assert_malformed(
            "\n<function=add>\nstray\n</function>\n",
            ToolCallError::UnexpectedText,
        );
    }

    #[test]
    fn text_after_the_function_close_is_reported() {
        assert_malformed(
            "\n<function=add>\n</function>\ntrailing\n",
            ToolCallError::UnexpectedText,
        );
    }

    #[test]
    fn a_repeated_parameter_name_is_reported() {
        assert_malformed(
            "\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=a>\n2\n</parameter>\n</function>\n",
            ToolCallError::DuplicateParameter("a".to_string()),
        );
    }

    #[test]
    fn a_call_that_never_closes_is_unterminated_at_finish() {
        let mut parser = parser(Start::Content);
        let first = parser.push("Sure.\n<tool_call>\n<function=add>\n<parameter=a>\n1");
        assert_eq!(first, [ToolEvent::Content("Sure.".to_string())], "push");
        let last = parser.finish();
        assert_eq!(
            last,
            [ToolEvent::Malformed(MalformedCall {
                error: ToolCallError::Unterminated,
                raw: "\n<function=add>\n<parameter=a>\n1".to_string(),
            })],
            "finish"
        );
    }

    #[test]
    fn a_call_marker_split_across_pieces_is_never_emitted() {
        let mut parser = parser(Start::Content);
        let first = parser.push("Hi <tool");
        assert_eq!(first, [ToolEvent::Content("Hi".to_string())], "piece 1");
        let second = parser.push("_call>\n<function=ping>\n</function>\n</tool");
        assert_eq!(second, [], "piece 2");
        let third = parser.push("_call> bye");
        assert_eq!(
            third,
            [
                ToolEvent::Call(call("call_0", "ping", "{}")),
                ToolEvent::Content("bye".to_string())
            ],
            "piece 3"
        );
        assert_eq!(parser.finish(), [], "finish");
    }

    #[test]
    fn a_think_close_marker_split_across_pieces_is_never_emitted() {
        let mut parser = parser(Start::Reasoning);
        let first = parser.push("plan</thi");
        assert_eq!(first, [ToolEvent::Reasoning("plan".to_string())], "piece 1");
        let second = parser.push("nk>ok");
        assert_eq!(second, [ToolEvent::Content("ok".to_string())], "piece 2");
        assert_eq!(parser.finish(), [], "finish");
    }

    #[test]
    fn a_lone_angle_bracket_is_released_once_it_cannot_start_a_marker() {
        let mut parser = parser(Start::Content);
        let first = parser.push("a <");
        assert_eq!(first, [ToolEvent::Content("a".to_string())], "piece 1");
        let second = parser.push("b");
        assert_eq!(second, [ToolEvent::Content(" <b".to_string())], "piece 2");
    }

    #[test]
    fn finish_releases_a_held_partial_marker_as_text() {
        let mut parser = parser(Start::Content);
        let first = parser.push("done <tool");
        assert_eq!(first, [ToolEvent::Content("done".to_string())], "push");
        let last = parser.finish();
        assert_eq!(last, [ToolEvent::Content(" <tool".to_string())], "finish");
    }

    #[test]
    fn a_think_block_then_content_then_a_call_in_one_buffer() {
        let text = concat!(
            "<think>a</think>b",
            "<tool_call>\n<function=run_python>\n",
            "<parameter=path>\nhello.py\n</parameter>\n",
            "</function>\n</tool_call>"
        );
        let expected = ParsedOutput {
            reasoning: "a".to_string(),
            content: "b".to_string(),
            tool_calls: vec![call("call_0", "run_python", r#"{"path":"hello.py"}"#)],
            malformed: Vec::new(),
        };
        assert_eq!(parse(text, Start::Content), expected, "one piece");
        let mut parser = parser(Start::Content);
        let mut events = Vec::new();
        for ch in text.chars() {
            events.extend(parser.push(&ch.to_string()));
        }
        events.extend(parser.finish());
        assert_eq!(
            ParsedOutput::from_events(events),
            expected,
            "one char per push"
        );
    }

    #[test]
    fn text_after_a_call_loses_its_leading_whitespace_and_follows_the_call() {
        let text = "<tool_call>\n<function=ping>\n</function>\n</tool_call>\n tail";
        let mut parser = parser(Start::Content);
        let mut events = parser.push(text);
        events.extend(parser.finish());
        assert_eq!(
            events,
            [
                ToolEvent::Call(call("call_0", "ping", "{}")),
                ToolEvent::Content("tail".to_string())
            ],
            "one piece"
        );
        assert_eq!(parse(text, Start::Content).content, "tail", "whole text");
    }

    #[test]
    fn streaming_one_character_at_a_time_gives_the_whole_text_result() {
        let text = concat!(
            "Need two files.\n</think>\n\nFirst I write.\n\n",
            "<tool_call>\n<function=write_file>\n",
            "<parameter=path>\nhello.py\n</parameter>\n",
            "<parameter=content>\nprint(2 + 3)\n</parameter>\n",
            "</function>\n</tool_call>\n",
            "<tool_call>\n<function=run_python>\n",
            "<parameter=path>\nhello.py\n</parameter>\n",
            "</function>\n</tool_call>"
        );
        let expected = ParsedOutput {
            reasoning: "Need two files.".to_string(),
            content: "First I write.".to_string(),
            tool_calls: vec![
                call(
                    "call_0",
                    "write_file",
                    r#"{"path":"hello.py","content":"print(2 + 3)"}"#,
                ),
                call("call_1", "run_python", r#"{"path":"hello.py"}"#),
            ],
            malformed: Vec::new(),
        };
        assert_eq!(parse(text, Start::Reasoning), expected, "whole text");
        let mut parser = parser(Start::Reasoning);
        let mut events = Vec::new();
        for ch in text.chars() {
            events.extend(parser.push(&ch.to_string()));
        }
        events.extend(parser.finish());
        for event in &events {
            if let ToolEvent::Reasoning(t) | ToolEvent::Content(t) = event {
                assert!(!t.contains('<'), "marker text leaked: {t:?}");
            }
        }
        assert_eq!(ParsedOutput::from_events(events), expected, "streamed");
    }

    #[test]
    fn the_pinned_template_output_parses_back_to_the_rendered_calls() {
        let mut template =
            ChatTemplate::from_str(include_str!("../../testdata/chat_template.jinja"))
                .expect("template");
        template.force_flags_off();
        let message = |role, content: &str, tool_calls| PromptMessage {
            role,
            content: Some(content.to_string()),
            reasoning_content: None,
            name: None,
            tool_call_id: None,
            tool_calls,
        };
        let spec = |name: &str, arguments: &str| ToolCallSpec {
            id: "unused".to_string(),
            name: name.to_string(),
            arguments: arguments.to_string(),
        };
        let messages = [
            message(Role::User, "Make it.", Vec::new()),
            message(
                Role::Assistant,
                "Checking.",
                vec![
                    spec(
                        "write_file",
                        r#"{"path": "a.py", "content": "print(1)\nprint(2)"}"#,
                    ),
                    spec(
                        "add",
                        r#"{"a": 5, "flag": true, "tags": ["x"], "label": "7"}"#,
                    ),
                ],
            ),
        ];
        let rendered = template
            .render(&messages, &[], Thinking::Off)
            .expect("render");
        assert!(
            rendered.contains(
                "<tool_call>\n<function=write_file>\n<parameter=path>\na.py\n</parameter>\n"
            ),
            "the pinned format changed: {rendered:?}"
        );
        let (_, tail) = rendered.split_once("</think>\n\n").expect("think block");
        let (text, _) = tail.split_once("<|im_end|>").expect("end of turn");
        let output = parse(text, Start::Content);
        assert_eq!(output.content, "Checking.", "content");
        assert_eq!(
            output.tool_calls,
            [
                call(
                    "call_0",
                    "write_file",
                    r#"{"path":"a.py","content":"print(1)\nprint(2)"}"#
                ),
                call(
                    "call_1",
                    "add",
                    r#"{"a":5,"flag":true,"tags":["x"],"label":"7"}"#
                ),
            ]
        );
    }

    /// Control spellings QUOTED in model prose stay text — the parser
    /// only honors the structural grammar, so a mid-text <parameter=...>
    /// or a bare <function=...> never opens a call. Mutation receipt: gut
    /// the grammar check (treat any spelling as a call open) and this pin
    /// must fail.
    #[test]
    fn quoted_control_spellings_in_prose_stay_plain_text() {
        let text = "the tag <parameter=x> and <function=add> are quoted, </think> too";
        let out = parse(text, Start::Content);
        assert!(out.tool_calls.is_empty(), "spelling opened a call: {out:?}");
        assert_eq!(
            out.content, text,
            "quoted spellings must survive byte-for-byte"
        );
        // A complete <tool_call> spelling in model output is the
        // structural opener by the upstream contract, even in prose.
        let quoting = "docs say: <tool_call><function=add><parameter=a>1</parameter></function></tool_call> ends here";
        let out = parse(quoting, Start::Content);
        assert_eq!(
            out.tool_calls.len(),
            1,
            "expected the quoted call to parse only if the grammar treats it as structural; got {out:?}"
        );
    }
}
