//! Validation of one chat completion request body.
//!
//! Contract: `ChatRequest` in `api/openapi.yaml`. The check covers the schema and the bounds.
//! Every failure is a `BadRequest` that names the field. Fields that the contract does not
//! define pass without a check, because the request schema permits extra client fields.

use super::error::ServerError;
use super::messages::{
    bad, optional_string, parse_messages, parse_tool_choice, parse_tools, required_string,
};
use crate::family::sampler::{SamplingDefaults, SamplingParams};
use crate::tokenizer::{PromptMessage, Thinking, ToolSpec};
use serde_json::{Map, Value};

/// The contract default for `max_tokens`.
const DEFAULT_MAX_TOKENS: u32 = 2_048;

/// The sampling fields of one request before the engine defaults apply: `None` means the request
/// omitted the field, and the values have passed the contract bounds check. The chat handler
/// resolves them against the defaults of the engine that serves the request.
#[derive(Debug)]
pub(crate) struct RequestedSampling {
    pub(crate) temperature: Option<f64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) top_k: Option<u64>,
}

/// A chat request that passed validation.
#[derive(Debug)]
pub(crate) struct ChatInput {
    pub(crate) model: Option<String>,
    pub(crate) engine: Option<String>,
    pub(crate) session: Option<String>,
    pub(crate) messages: Vec<PromptMessage>,
    pub(crate) tools: Vec<ToolSpec>,
    pub(crate) thinking: Thinking,
    pub(crate) stops: Vec<String>,
    pub(crate) sampling: RequestedSampling,
    pub(crate) max_tokens: u32,
    pub(crate) stream: bool,
}

/// Parses and validates a request body. The caller has already checked the size cap.
pub(crate) fn parse_chat(body: &[u8]) -> Result<ChatInput, ServerError> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|error| bad(format!("request body is not valid JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| bad("request body must be a JSON object"))?;
    let messages = parse_messages(object.get("messages"))?;
    let mut tools = parse_tools(object.get("tools"))?;
    if parse_tool_choice(object.get("tool_choice"))? {
        tools.clear();
    }
    check_constrained_output(object)?;
    let sampling = RequestedSampling {
        temperature: number(object, "temperature")?,
        top_p: number(object, "top_p")?,
        top_k: count(object, "top_k", 0)?,
    };
    // The bounds of the fields the request sets are checked here, before the engine is known.
    // The defaults of the engine that serves the request apply in the chat handler.
    SamplingParams::from_request(
        sampling.temperature,
        sampling.top_p,
        sampling.top_k,
        &SamplingDefaults::CONTRACT,
    )?;
    let max_tokens = match count(object, "max_tokens", 1)? {
        None => DEFAULT_MAX_TOKENS,
        Some(limit) => u32::try_from(limit).unwrap_or(u32::MAX),
    };
    Ok(ChatInput {
        model: optional_string(object, "model", "")?,
        engine: optional_string(object, "x-engine", "")?,
        session: optional_string(object, "x-session", "")?,
        messages,
        tools,
        thinking: thinking(object)?,
        stops: stops(object)?,
        sampling,
        max_tokens,
        stream: flag(object, "stream")?,
    })
}

/// The one place that rejects output controls which need grammar-constrained decoding.
/// A `response_format` of type `json_object` or `json_schema`, `tool_choice` `required`, and a
/// named `tool_choice` return `BadRequest` until the grammar slice replaces this function.
/// `response_format` of type `text` and `tool_choice` `auto` or `none` pass. The shape of every
/// member is checked first, so a malformed member gets its own message.
fn check_constrained_output(object: &Map<String, Value>) -> Result<(), ServerError> {
    match object.get("response_format") {
        None => {}
        Some(Value::Object(format)) => {
            match optional_string(format, "type", "response_format.")?.as_deref() {
                None => return Err(bad("response_format.type is required")),
                Some("text") => {}
                Some("json_object") => {
                    return Err(grammar_unavailable("response_format of type json_object"));
                }
                Some("json_schema") => {
                    check_json_schema_shape(format)?;
                    return Err(grammar_unavailable("response_format of type json_schema"));
                }
                Some(_) => {
                    return Err(bad(
                        "response_format.type must be text, json_object, or json_schema",
                    ));
                }
            }
        }
        Some(_) => return Err(bad("response_format must be an object")),
    }
    match object.get("tool_choice") {
        Some(Value::String(choice)) if choice == "required" => {
            Err(grammar_unavailable("tool_choice required"))
        }
        Some(Value::Object(_)) => Err(grammar_unavailable("a named tool_choice")),
        _ => Ok(()),
    }
}

fn grammar_unavailable(what: &str) -> ServerError {
    bad(format!(
        "{what} needs grammar-constrained decoding, which is not available yet"
    ))
}

/// Checks `response_format.json_schema`: a `name` string, an optional `schema` object, and an
/// optional boolean `strict`.
fn check_json_schema_shape(format: &Map<String, Value>) -> Result<(), ServerError> {
    let schema = format
        .get("json_schema")
        .and_then(Value::as_object)
        .ok_or_else(|| bad("response_format.json_schema must be an object"))?;
    required_string(schema, "name", "response_format.json_schema.")?;
    if schema
        .get("schema")
        .is_some_and(|schema| !schema.is_object())
    {
        return Err(bad("response_format.json_schema.schema must be an object"));
    }
    if schema
        .get("strict")
        .is_some_and(|strict| !strict.is_boolean())
    {
        return Err(bad("response_format.json_schema.strict must be a boolean"));
    }
    Ok(())
}

/// Reads an optional number member.
fn number(object: &Map<String, Value>, field: &str) -> Result<Option<f64>, ServerError> {
    match object.get(field) {
        None => Ok(None),
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
        None => Ok(None),
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
        None => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(bad(format!("{field} must be a boolean"))),
    }
}

/// Reads `x-thinking`: `default`, `on`, or `off`.
fn thinking(object: &Map<String, Value>) -> Result<Thinking, ServerError> {
    match optional_string(object, "x-thinking", "")?.as_deref() {
        None | Some("default") => Ok(Thinking::Default),
        Some("on") => Ok(Thinking::On),
        Some("off") => Ok(Thinking::Off),
        Some(_) => Err(bad("x-thinking must be default, on, or off")),
    }
}

/// Reads `stop`: an array of strings.
fn stops(object: &Map<String, Value>) -> Result<Vec<String>, ServerError> {
    let Some(value) = object.get("stop") else {
        return Ok(Vec::new());
    };
    let items = value
        .as_array()
        .ok_or_else(|| bad("stop must be an array of strings"))?;
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            item.as_str()
                .map(str::to_string)
                .ok_or_else(|| bad(format!("stop[{index}] must be a string")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{ChatInput, parse_chat};
    use crate::EngError;
    use crate::family::sampler::{SamplingDefaults, SamplingParams};
    use crate::server::error::ServerError;
    use crate::tokenizer::Thinking;
    use serde_json::{Value, json};

    fn user(text: &str) -> Value {
        json!([{"role": "user", "content": text}])
    }

    fn parse(body: &Value) -> Result<ChatInput, ServerError> {
        parse_chat(body.to_string().as_bytes())
    }

    fn rejected(body: &Value) -> String {
        match parse(body).expect_err("must be rejected") {
            ServerError::Eng(EngError::BadRequest(message)) => message,
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn a_minimal_request_takes_the_contract_defaults() {
        let input = parse(&json!({"model": "m", "messages": user("hi")})).expect("valid");
        assert_eq!(input.model.as_deref(), Some("m"));
        assert_eq!(input.engine, None);
        assert_eq!(input.session, None);
        assert_eq!(input.max_tokens, 2_048);
        assert!(!input.stream);
        assert!(matches!(input.thinking, Thinking::Default));
        assert!(input.stops.is_empty());
        assert!(input.tools.is_empty());
        assert_eq!(input.sampling.temperature, None);
        assert_eq!(input.sampling.top_p, None);
        assert_eq!(input.sampling.top_k, None);
        assert_eq!(input.messages.len(), 1);
    }

    #[test]
    fn every_defined_field_reaches_the_input() {
        let body = json!({
            "model": "m", "x-engine": "e2", "x-session": "s1",
            "messages": user("hi"), "stream": true, "max_tokens": 16,
            "temperature": 0.25, "top_p": 0.5, "top_k": 40,
            "stop": ["END", "\n\n"], "x-thinking": "on",
            "tools": [{"type": "function", "function": {"name": "add"}}],
            "tool_choice": "auto"
        });
        let input = parse(&body).expect("valid");
        assert_eq!(input.engine.as_deref(), Some("e2"));
        assert_eq!(input.session.as_deref(), Some("s1"));
        assert_eq!(input.max_tokens, 16);
        assert!(input.stream);
        assert_eq!(input.stops, ["END", "\n\n"]);
        assert!(matches!(input.thinking, Thinking::On));
        assert_eq!(input.tools.len(), 1);
        assert_eq!(input.sampling.temperature, Some(0.25));
        assert_eq!(input.sampling.top_p, Some(0.5));
        assert_eq!(input.sampling.top_k, Some(40));
        assert_eq!(
            SamplingParams::from_request(
                input.sampling.temperature,
                input.sampling.top_p,
                input.sampling.top_k,
                &SamplingDefaults::CONTRACT
            ),
            SamplingParams::from_request(
                Some(0.25),
                Some(0.5),
                Some(40),
                &SamplingDefaults::CONTRACT
            )
        );
    }

    #[test]
    fn tool_choice_none_removes_the_tools() {
        let body = json!({
            "model": "m", "messages": user("hi"),
            "tools": [{"type": "function", "function": {"name": "add"}}],
            "tool_choice": "none"
        });
        assert!(parse(&body).expect("valid").tools.is_empty());
    }

    #[test]
    fn a_body_that_is_not_json_or_not_an_object_is_rejected() {
        let error = parse_chat(b"{\"messages\":").expect_err("truncated");
        assert!(matches!(error, ServerError::Eng(EngError::BadRequest(_))));
        assert!(rejected(&json!([1, 2])).contains("JSON object"));
    }

    #[test]
    fn a_body_nested_beyond_the_recursion_limit_is_rejected() {
        let nested = format!("{}1{}", "[".repeat(200), "]".repeat(200));
        let body = format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"x"}}],"tools":{nested}}}"#
        );
        let error = parse_chat(body.as_bytes()).expect_err("too deep");
        match error {
            ServerError::Eng(EngError::BadRequest(message)) => {
                assert!(message.contains("recursion limit"), "{message}");
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn max_tokens_below_1_names_field() {
        for bad in [json!(0), json!(-3), json!(1.5), json!("8"), json!(null)] {
            let message =
                rejected(&json!({"model": "m", "messages": user("hi"), "max_tokens": bad}));
            assert!(message.contains("max_tokens"), "{bad}: {message}");
        }
    }

    #[test]
    fn a_huge_max_tokens_saturates_at_the_u32_limit() {
        let body = json!({"model": "m", "messages": user("hi"), "max_tokens": 1_u64 << 40});
        assert_eq!(parse(&body).expect("valid").max_tokens, u32::MAX);
    }

    #[test]
    fn temperature_out_of_range_returns_badrequest() {
        let low = rejected(&json!({"model": "m", "messages": user("hi"), "temperature": -0.1}));
        assert!(low.contains("temperature"), "{low}");
        let high = rejected(&json!({"model": "m", "messages": user("hi"), "temperature": 2.5}));
        assert!(high.contains("temperature"), "{high}");
        let kind = rejected(&json!({"model": "m", "messages": user("hi"), "temperature": "hot"}));
        assert!(kind.contains("temperature"), "{kind}");
    }

    #[test]
    fn top_p_out_of_range_names_the_field() {
        let message = rejected(&json!({"model": "m", "messages": user("hi"), "top_p": 1.5}));
        assert!(message.contains("top_p"), "{message}");
    }

    #[test]
    fn top_k_must_be_a_whole_number_from_zero() {
        for bad in [json!(-1), json!(2.5), json!("4")] {
            let message = rejected(&json!({"model": "m", "messages": user("hi"), "top_k": bad}));
            assert!(message.contains("top_k"), "{bad}: {message}");
        }
        let zero = parse(&json!({"model": "m", "messages": user("hi"), "top_k": 0}));
        assert!(zero.is_ok(), "top_k 0 turns the filter off");
    }

    #[test]
    fn a_null_sampling_field_in_a_request_stays_a_bad_request() {
        // The file loader treats an explicit null as an absent key, but a request field stays
        // checked against the contract: a null is a wrong type and fails with 400.
        for field in ["temperature", "top_p", "top_k"] {
            let body = json!({"model": "m", "messages": user("hi"), field: null});
            let message = rejected(&body);
            assert!(message.contains(field), "{field}: {message}");
        }
    }

    #[test]
    fn stream_stop_and_thinking_check_their_types() {
        let stream = rejected(&json!({"model": "m", "messages": user("hi"), "stream": "yes"}));
        assert!(stream.contains("stream"), "{stream}");
        let stop = rejected(&json!({"model": "m", "messages": user("hi"), "stop": "END"}));
        assert!(stop.contains("stop"), "{stop}");
        let item = rejected(&json!({"model": "m", "messages": user("hi"), "stop": ["a", 3]}));
        assert!(item.contains("stop[1]"), "{item}");
        let mode = rejected(&json!({"model": "m", "messages": user("hi"), "x-thinking": "maybe"}));
        assert!(mode.contains("x-thinking"), "{mode}");
    }

    #[test]
    fn thinking_off_and_default_convert() {
        let off = parse(&json!({"model": "m", "messages": user("hi"), "x-thinking": "off"}));
        assert!(matches!(off.expect("valid").thinking, Thinking::Off));
        let default =
            parse(&json!({"model": "m", "messages": user("hi"), "x-thinking": "default"}));
        assert!(matches!(
            default.expect("valid").thinking,
            Thinking::Default
        ));
    }

    #[test]
    fn routing_fields_must_be_strings() {
        for field in ["model", "x-engine", "x-session"] {
            let message = rejected(&json!({"messages": user("hi"), field: 7}));
            assert!(message.contains(field), "{field}: {message}");
        }
    }

    #[test]
    fn fields_outside_the_contract_pass_without_a_check() {
        let body = json!({"model": "m", "messages": user("hi"), "n": 1, "user": "u", "seed": 3});
        assert!(parse(&body).is_ok());
    }

    fn grammar_message(body: &Value) -> String {
        let message = rejected(body);
        assert!(
            message.contains("grammar-constrained decoding"),
            "the message must say why: {message}"
        );
        message
    }

    #[test]
    fn response_format_text_and_tool_choice_auto_are_accepted() {
        let body = json!({
            "model": "m", "messages": user("hi"),
            "response_format": {"type": "text"},
            "tools": [{"type": "function", "function": {"name": "add", "strict": true}}],
            "tool_choice": "auto"
        });
        assert!(parse(&body).is_ok());
    }

    #[test]
    fn response_format_json_object_returns_400_that_names_the_grammar() {
        let body = json!({"model": "m", "messages": user("hi"), "response_format": {"type": "json_object"}});
        assert!(grammar_message(&body).contains("json_object"));
    }

    #[test]
    fn response_format_json_schema_returns_400_that_names_the_grammar() {
        let body = json!({
            "model": "m", "messages": user("hi"),
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "reply", "schema": {"type": "object"}, "strict": true}}
        });
        assert!(grammar_message(&body).contains("json_schema"));
    }

    #[test]
    fn a_json_schema_format_without_a_schema_member_is_still_gated() {
        let body = json!({
            "model": "m", "messages": user("hi"),
            "response_format": {"type": "json_schema", "json_schema": {"name": "reply"}}
        });
        assert!(grammar_message(&body).contains("json_schema"));
    }

    #[test]
    fn tool_choice_required_returns_400_that_names_the_grammar() {
        let body = json!({
            "model": "m", "messages": user("hi"),
            "tools": [{"type": "function", "function": {"name": "add"}}],
            "tool_choice": "required"
        });
        assert!(grammar_message(&body).contains("tool_choice required"));
    }

    #[test]
    fn a_named_tool_choice_returns_400_that_names_the_grammar() {
        let body = json!({
            "model": "m", "messages": user("hi"),
            "tools": [{"type": "function", "function": {"name": "add"}}],
            "tool_choice": {"type": "function", "function": {"name": "add"}}
        });
        assert!(grammar_message(&body).contains("named tool_choice"));
    }

    #[test]
    fn a_malformed_response_format_gets_its_own_message() {
        let cases = [
            (json!("json"), "response_format must be an object"),
            (json!({}), "response_format.type is required"),
            (json!({"type": "xml"}), "response_format.type must be"),
            (
                json!({"type": "json_schema"}),
                "response_format.json_schema must be an object",
            ),
            (
                json!({"type": "json_schema", "json_schema": {"schema": {}}}),
                "json_schema.name",
            ),
            (
                json!({"type": "json_schema", "json_schema": {"name": "n", "schema": 3}}),
                "json_schema.schema",
            ),
            (
                json!({"type": "json_schema", "json_schema": {"name": "n", "schema": {}, "strict": "yes"}}),
                "json_schema.strict",
            ),
        ];
        for (format, expected) in cases {
            let message =
                rejected(&json!({"model": "m", "messages": user("hi"), "response_format": format}));
            assert!(message.contains(expected), "{format}: {message}");
            assert!(
                !message.contains("grammar-constrained"),
                "{format}: {message}"
            );
        }
    }

    #[test]
    fn the_contract_schema_agrees_with_the_response_format_rules() {
        use crate::server::contract::matches_schema;
        let schema_format =
            json!({"type": "json_schema", "json_schema": {"name": "n", "schema": {}}});
        for format in [
            json!({"type": "text"}),
            json!({"type": "json_object"}),
            schema_format,
        ] {
            let body = json!({"model": "m", "messages": user("hi"), "response_format": format});
            assert!(matches_schema("ChatRequest", &body), "{format}");
        }
        for format in [
            json!({"type": "xml"}),
            json!({"type": "json_schema"}),
            json!("json"),
        ] {
            let body = json!({"model": "m", "messages": user("hi"), "response_format": format});
            assert!(!matches_schema("ChatRequest", &body), "{format}");
        }
        let strict = json!([{"type": "function", "function": {"name": "f", "strict": "yes"}}]);
        let body = json!({"model": "m", "messages": user("hi"), "tools": strict});
        assert!(!matches_schema("ChatRequest", &body));
    }

    #[test]
    fn a_message_error_reaches_the_caller_with_its_field_path() {
        let body = json!({"model": "m", "messages": [{"role": "user"}]});
        assert!(rejected(&body).contains("messages[0].content"));
    }
}
