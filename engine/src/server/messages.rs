//! Message and tool validation for chat requests.
//!
//! Contract: `ChatMessage`, `ToolCall`, `Tool`, and `ToolChoice` in `api/openapi.yaml`, and the
//! section "Tool requests and message roles" of the server design.
//! Every failure is a `BadRequest` that names the field.
//!
//! `name` is not a member of `ChatMessage`, and the contract permits extra members.
//! A string `name` reaches the template. A `name` of any other type is ignored.
//!
//! `content` is a string or an array of text parts. The parts are joined in order
//! with no separator, so the template sees one string.

use super::error::ServerError;
use crate::EngError;
use crate::tokenizer::{PromptMessage, Role, ToolCallSpec, ToolSpec};
use serde_json::{Map, Value};

/// A `BadRequest` with the given message.
pub(super) fn bad(message: impl Into<String>) -> ServerError {
    ServerError::Eng(EngError::BadRequest(message.into()))
}

/// Reads an optional string member. An absent member gives `None`. Any other type fails.
pub(super) fn optional_string(
    object: &Map<String, Value>,
    field: &str,
    at: &str,
) -> Result<Option<String>, ServerError> {
    match object.get(field) {
        None => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => Err(bad(format!("{at}{field} must be a string"))),
    }
}

/// Reads a required string member.
pub(super) fn required_string(
    object: &Map<String, Value>,
    field: &str,
    at: &str,
) -> Result<String, ServerError> {
    optional_string(object, field, at)?.ok_or_else(|| bad(format!("{at}{field} is required")))
}

/// Validates the `messages` array and converts it to the template input.
pub(super) fn parse_messages(value: Option<&Value>) -> Result<Vec<PromptMessage>, ServerError> {
    let items = value
        .ok_or_else(|| bad("messages is required"))?
        .as_array()
        .ok_or_else(|| bad("messages must be an array"))?;
    if items.is_empty() {
        return Err(bad("messages must not be empty"));
    }
    items
        .iter()
        .enumerate()
        .map(|(index, item)| parse_message(index, item))
        .collect()
}

fn parse_message(index: usize, item: &Value) -> Result<PromptMessage, ServerError> {
    let at = format!("messages[{index}].");
    let object = item
        .as_object()
        .ok_or_else(|| bad(format!("messages[{index}] must be an object")))?;
    let role = match required_string(object, "role", &at)?.as_str() {
        "system" => Role::System,
        "developer" => Role::Developer,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        _ => {
            return Err(bad(format!(
                "{at}role must be system, developer, user, assistant, or tool"
            )));
        }
    };
    let content = match object.get("content") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(Value::Array(parts)) => Some(join_parts(parts, &at)?),
        Some(_) => return Err(bad(format!("{at}content must be a string or an array"))),
    };
    let tool_calls = parse_tool_calls(object.get("tool_calls"), &at)?;
    let tool_call_id = optional_string(object, "tool_call_id", &at)?;
    match role {
        Role::Tool => {
            if tool_call_id.is_none() {
                return Err(bad(format!(
                    "{at}tool_call_id is required for a tool message"
                )));
            }
            if content.is_none() {
                return Err(bad(format!("{at}content is required for a tool message")));
            }
        }
        Role::System | Role::Developer | Role::User => {
            if content.is_none() {
                return Err(bad(format!("{at}content is required")));
            }
        }
        Role::Assistant => {
            if content.is_none() && !object.contains_key("tool_calls") {
                return Err(bad(format!(
                    "{at}content is required unless tool_calls is present"
                )));
            }
        }
    }
    Ok(PromptMessage {
        role,
        content,
        reasoning_content: object
            .get("reasoning_content")
            .and_then(Value::as_str)
            .map(str::to_string),
        name: object
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string),
        tool_call_id,
        tool_calls,
    })
}

// Joins the text of an array of content parts. Only text parts are supported; every
// other shape names the message index, the part index, and the broken member.
fn join_parts(parts: &[Value], at: &str) -> Result<String, ServerError> {
    let mut joined = String::new();
    for (part_index, part) in parts.iter().enumerate() {
        let object = part
            .as_object()
            .ok_or_else(|| bad(format!("{at}content[{part_index}] must be an object")))?;
        match object.get("type").and_then(Value::as_str) {
            Some("text") => {}
            _ => return Err(bad(format!("{at}content[{part_index}].type must be text"))),
        }
        match object.get("text") {
            Some(Value::String(piece)) => joined.push_str(piece),
            _ => {
                return Err(bad(format!(
                    "{at}content[{part_index}].text must be a string"
                )));
            }
        }
    }
    Ok(joined)
}

fn parse_tool_calls(value: Option<&Value>, at: &str) -> Result<Vec<ToolCallSpec>, ServerError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let items = value
        .as_array()
        .ok_or_else(|| bad(format!("{at}tool_calls must be an array")))?;
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let call_at = format!("{at}tool_calls[{index}].");
            let object = item
                .as_object()
                .ok_or_else(|| bad(format!("{at}tool_calls[{index}] must be an object")))?;
            if required_string(object, "type", &call_at)? != "function" {
                return Err(bad(format!("{call_at}type must be function")));
            }
            let function = object
                .get("function")
                .and_then(Value::as_object)
                .ok_or_else(|| bad(format!("{call_at}function must be an object")))?;
            let function_at = format!("{call_at}function.");
            Ok(ToolCallSpec {
                id: required_string(object, "id", &call_at)?,
                name: required_string(function, "name", &function_at)?,
                arguments: required_string(function, "arguments", &function_at)?,
            })
        })
        .collect()
}

/// Validates `tools`. Each element passes to the template unchanged.
pub(super) fn parse_tools(value: Option<&Value>) -> Result<Vec<ToolSpec>, ServerError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let items = value
        .as_array()
        .ok_or_else(|| bad("tools must be an array"))?;
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let at = format!("tools[{index}].");
            let object = item
                .as_object()
                .ok_or_else(|| bad(format!("tools[{index}] must be an object")))?;
            if required_string(object, "type", &at)? != "function" {
                return Err(bad(format!("{at}type must be function")));
            }
            let function = object
                .get("function")
                .and_then(Value::as_object)
                .ok_or_else(|| bad(format!("{at}function must be an object")))?;
            required_string(function, "name", &format!("{at}function."))?;
            if function
                .get("description")
                .is_some_and(|description| !description.is_string())
            {
                return Err(bad(format!("{at}function.description must be a string")));
            }
            if function
                .get("parameters")
                .is_some_and(|parameters| !parameters.is_object())
            {
                return Err(bad(format!("{at}function.parameters must be an object")));
            }
            if function
                .get("strict")
                .is_some_and(|strict| !strict.is_boolean())
            {
                return Err(bad(format!("{at}function.strict must be a boolean")));
            }
            Ok(ToolSpec(item.clone()))
        })
        .collect()
}

/// Validates `tool_choice`. It returns true when the choice is `none`, which disables the tools.
pub(super) fn parse_tool_choice(value: Option<&Value>) -> Result<bool, ServerError> {
    match value {
        None => Ok(false),
        Some(Value::String(choice)) => match choice.as_str() {
            "none" => Ok(true),
            "auto" | "required" => Ok(false),
            _ => Err(bad(
                "tool_choice must be none, auto, required, or a function object",
            )),
        },
        Some(Value::Object(object)) => {
            if required_string(object, "type", "tool_choice.")? != "function" {
                return Err(bad("tool_choice.type must be function"));
            }
            let function = object
                .get("function")
                .and_then(Value::as_object)
                .ok_or_else(|| bad("tool_choice.function must be an object"))?;
            required_string(function, "name", "tool_choice.function.")?;
            Ok(false)
        }
        Some(_) => Err(bad(
            "tool_choice must be none, auto, required, or a function object",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_messages, parse_tool_choice, parse_tools};
    use crate::EngError;
    use crate::server::contract::matches_schema;
    use crate::server::error::ServerError;
    use serde_json::{Value, json};

    fn messages(value: Value) -> Result<Vec<crate::tokenizer::PromptMessage>, ServerError> {
        parse_messages(Some(&value))
    }

    fn message_of(error: ServerError) -> String {
        match error {
            ServerError::Eng(EngError::BadRequest(message)) => message,
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    fn rejected(value: Value) -> String {
        message_of(messages(value).expect_err("must be rejected"))
    }

    /// One text content part.
    fn text_part(text: &str) -> Value {
        json!({"type": "text", "text": text})
    }

    #[test]
    fn the_four_roles_convert_in_order() {
        let converted = messages(json!([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": "t", "tool_call_id": "c1"}
        ]))
        .expect("valid messages");
        let roles: Vec<String> = converted.iter().map(|m| format!("{:?}", m.role)).collect();
        assert_eq!(roles, ["System", "User", "Assistant", "Tool"]);
        assert_eq!(converted[0].content.as_deref(), Some("s"));
        assert_eq!(converted[3].tool_call_id.as_deref(), Some("c1"));
    }

    #[test]
    fn the_name_member_reaches_the_template_input() {
        let converted = messages(json!([{"role": "user", "content": "u", "name": "dan"}]))
            .expect("valid messages");
        assert_eq!(converted[0].name.as_deref(), Some("dan"));
    }

    #[test]
    fn a_name_that_is_not_a_string_is_ignored() {
        let converted = messages(json!([{"role": "user", "content": "u", "name": 5}]))
            .expect("the contract permits extra members");
        assert!(converted[0].name.is_none());
    }

    #[test]
    fn a_tool_call_id_that_is_not_a_string_names_the_field() {
        let message = rejected(json!([{"role": "user", "content": "u", "tool_call_id": 5}]));
        assert!(message.contains("messages[0].tool_call_id"), "{message}");
    }

    #[test]
    fn tool_calls_that_are_not_an_array_name_the_field() {
        let message = rejected(json!([{"role": "assistant", "content": "a", "tool_calls": "x"}]));
        assert!(message.contains("messages[0].tool_calls"), "{message}");
    }

    #[test]
    fn the_converter_accepts_exactly_the_messages_the_schema_accepts() {
        let call =
            json!({"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}});
        let cases = [
            json!({"role": "system", "content": "s"}),
            json!({"role": "developer", "content": "d"}),
            json!({"role": "developer", "content": [text_part("d")]}),
            json!({"role": "developer"}),
            json!({"role": "user", "content": "u", "name": "dan", "x-extra": 1}),
            json!({"role": "user", "content": "u", "name": 5}),
            json!({"role": "assistant", "content": "a"}),
            json!({"role": "assistant", "content": null, "tool_calls": [call]}),
            json!({"role": "assistant", "content": null, "tool_calls": []}),
            json!({"role": "tool", "content": "42", "tool_call_id": "c"}),
            json!({"role": "user"}),
            json!({"role": "user", "content": null}),
            json!({"role": "user", "content": [{"type": "text", "text": "x"}]}),
            json!({"role": "user", "content": []}),
            json!({"role": "user", "content": [text_part("a"), text_part("b"), {"x-extra": 1}]}),
            json!({"role": "system", "content": [text_part("s")]}),
            json!({"role": "assistant", "content": [text_part("a")], "tool_calls": [call]}),
            json!({"role": "tool", "content": [text_part("42")], "tool_call_id": "c"}),
            json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}),
            json!({"role": "user", "content": [{"type": "text"}]}),
            json!({"role": "user", "content": [{"type": "text", "text": 5}]}),
            json!({"role": "user", "content": ["x"]}),
            json!({"role": "assistant", "content": null}),
            json!({"role": "assistant"}),
            json!({"role": "tool", "content": "42"}),
            json!({"role": "tool", "tool_call_id": "c"}),
            json!({"role": "user", "content": "u", "tool_call_id": 5}),
            json!({"role": "assistant", "content": "a", "tool_calls": "x"}),
            json!({"role": "robot", "content": "x"}),
            json!({"content": "x"}),
        ];
        for case in cases {
            assert_eq!(
                messages(json!([case])).is_ok(),
                matches_schema("ChatMessage", &case),
                "converter and ChatMessage schema disagree on {case}"
            );
        }
    }

    #[test]
    fn a_missing_empty_or_mistyped_message_list_is_rejected() {
        assert!(message_of(parse_messages(None).expect_err("missing")).contains("messages"));
        assert!(rejected(json!([])).contains("not be empty"));
        assert!(rejected(json!("hi")).contains("array"));
        assert!(rejected(json!(["hi"])).contains("messages[0]"));
    }

    #[test]
    fn an_unknown_role_names_the_field() {
        let message =
            rejected(json!([{"role": "user", "content": "u"}, {"role": "robot", "content": "x"}]));
        assert!(message.contains("messages[1].role"), "{message}");
    }

    #[test]
    fn a_capitalized_developer_role_is_rejected() {
        let message = rejected(json!([{"role": "Developer", "content": "d"}]));
        assert!(message.contains("messages[0].role"), "{message}");
    }

    #[test]
    fn a_developer_message_preserves_its_role() {
        let converted = messages(json!([
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": "hi"}
        ]))
        .expect("valid messages");
        assert_eq!(format!("{:?}", converted[0].role), "Developer");
        assert_eq!(converted[0].content.as_deref(), Some("be brief"));
        let unknown = rejected(json!([
            {"role": "developer", "content": [text_part("brief")]},
            {"role": "robot", "content": "x"}
        ]));
        assert!(
            unknown.contains("role must be system, developer, user, assistant, or tool"),
            "{unknown}"
        );
    }

    #[test]
    fn a_missing_role_names_the_field() {
        let message = rejected(json!([{"content": "x"}]));
        assert!(message.contains("messages[0].role"), "{message}");
    }

    #[test]
    fn string_content_reaches_the_template_unchanged() {
        let converted =
            messages(json!([{"role": "user", "content": "plain text"}])).expect("valid messages");
        assert_eq!(converted[0].content.as_deref(), Some("plain text"));
    }

    #[test]
    fn text_parts_are_concatenated_in_order_with_no_separator() {
        let converted = messages(json!([{
            "role": "user",
            "content": [
                text_part("You "),
                text_part("add."),
                text_part(" Now.")
            ]
        }]))
        .expect("valid messages");
        assert_eq!(converted[0].content.as_deref(), Some("You add. Now."));
    }

    #[test]
    fn an_empty_content_array_gives_an_empty_string() {
        let converted = messages(json!([{"role": "user", "content": []}])).expect("valid messages");
        assert_eq!(converted[0].content.as_deref(), Some(""));
    }

    #[test]
    fn an_assistant_message_can_mix_array_content_and_tool_calls() {
        let converted = messages(json!([{
            "role": "assistant",
            "content": [text_part("Let me "), text_part("check.")],
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": "{\"a\":1}"}
            }]
        }]))
        .expect("valid messages");
        assert_eq!(converted[0].content.as_deref(), Some("Let me check."));
        assert_eq!(converted[0].tool_calls[0].name, "add");
        assert_eq!(converted[0].tool_calls[0].id, "call_1");
    }

    #[test]
    fn a_tool_message_can_carry_array_content() {
        let converted = messages(json!([{
            "role": "tool",
            "content": [text_part("sum is "), text_part("3")],
            "tool_call_id": "call_1"
        }]))
        .expect("valid messages");
        assert_eq!(format!("{:?}", converted[0].role), "Tool");
        assert_eq!(converted[0].content.as_deref(), Some("sum is 3"));
    }

    #[test]
    fn a_tool_message_with_array_content_needs_a_tool_call_id() {
        let message = rejected(json!([{"role": "tool", "content": [text_part("42")]}]));
        assert_eq!(
            message,
            "messages[0].tool_call_id is required for a tool message"
        );
    }

    #[test]
    fn a_developer_message_with_array_content_preserves_its_role() {
        let converted = messages(json!([{
            "role": "developer",
            "content": [text_part("be "), text_part("brief")]
        }]))
        .expect("valid messages");
        assert_eq!(format!("{:?}", converted[0].role), "Developer");
        assert_eq!(converted[0].content.as_deref(), Some("be brief"));
    }

    #[test]
    fn an_unsupported_part_type_names_the_message_and_part_index() {
        let message = rejected(json!([
            {"role": "user", "content": "hi"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
            ]}
        ]));
        assert_eq!(message, "messages[1].content[0].type must be text");
    }

    #[test]
    fn a_long_unsupported_part_type_is_not_echoed_in_the_error() {
        let long_type = "x".repeat(100_000);
        let message =
            rejected(json!([{"role": "user", "content": [{"type": long_type, "text": "t"}]}]));
        assert!(
            message.len() < 200,
            "error message is {} bytes: {message}",
            message.len()
        );
        assert!(message.contains("messages[0].content[0]"), "{message}");
    }

    #[test]
    fn a_null_or_numeric_part_type_gives_the_must_be_text_message() {
        let null_type = rejected(json!([
            {"role": "user", "content": [{"type": null, "text": "t"}]}
        ]));
        assert_eq!(null_type, "messages[0].content[0].type must be text");
        let numeric_type = rejected(json!([
            {"role": "user", "content": [{"type": 5, "text": "t"}]}
        ]));
        assert_eq!(numeric_type, "messages[0].content[0].type must be text");
    }

    #[test]
    fn a_text_part_without_a_string_text_is_rejected() {
        let wrong_type = rejected(json!([
            {"role": "user", "content": [text_part("ok"), {"type": "text", "text": 5}]}
        ]));
        assert!(
            wrong_type.contains("messages[0].content[1].text"),
            "{wrong_type}"
        );
        let missing = rejected(json!([{"role": "user", "content": [{"type": "text"}]}]));
        assert!(missing.contains("messages[0].content[0].text"), "{missing}");
    }

    #[test]
    fn a_part_that_is_not_an_object_is_rejected() {
        let message = rejected(json!([{"role": "user", "content": ["plain text"]}]));
        assert!(
            message.contains("messages[0].content[0] must be an object"),
            "{message}"
        );
    }

    #[test]
    fn system_and_user_content_must_be_present_and_not_null() {
        let missing = rejected(json!([{"role": "user"}]));
        assert!(missing.contains("messages[0].content"), "{missing}");
        let null = rejected(json!([{"role": "system", "content": null}]));
        assert!(null.contains("messages[0].content"), "{null}");
        let number = rejected(json!([{"role": "user", "content": 5}]));
        assert!(number.contains("messages[0].content"), "{number}");
    }

    #[test]
    fn tool_message_without_tool_call_id_rejected() {
        let message = rejected(json!([{"role": "tool", "content": "42"}]));
        assert!(message.contains("tool_call_id"), "{message}");
    }

    #[test]
    fn a_tool_message_needs_string_content() {
        let message = rejected(json!([{"role": "tool", "tool_call_id": "c1"}]));
        assert!(message.contains("messages[0].content"), "{message}");
    }

    #[test]
    fn assistant_message_without_content_and_tool_calls_rejected() {
        let message = rejected(json!([{"role": "assistant", "content": null}]));
        assert!(message.contains("messages[0].content"), "{message}");
        let absent = rejected(json!([{"role": "assistant"}]));
        assert!(absent.contains("tool_calls"), "{absent}");
    }

    #[test]
    fn an_assistant_message_with_tool_calls_may_have_null_content() {
        let converted = messages(json!([{
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": "{\"a\":1}"}
            }]
        }]))
        .expect("valid messages");
        assert_eq!(format!("{:?}", converted[0].role), "Assistant");
        assert!(converted[0].content.is_none());
        let call = &converted[0].tool_calls[0];
        assert_eq!(call.id, "call_1");
        assert_eq!(call.name, "add");
        assert_eq!(call.arguments, "{\"a\":1}");
    }

    #[test]
    fn a_tool_call_names_its_broken_member() {
        let call = |function: Value, kind: &str| json!([{"role": "assistant", "tool_calls": [{"id": "c", "type": kind, "function": function}]}]);
        let kind = rejected(call(json!({"name": "f", "arguments": "{}"}), "plugin"));
        assert!(kind.contains("messages[0].tool_calls[0].type"), "{kind}");
        let name = rejected(call(json!({"arguments": "{}"}), "function"));
        assert!(
            name.contains("messages[0].tool_calls[0].function.name"),
            "{name}"
        );
        let arguments = rejected(call(
            json!({"name": "f", "arguments": {"a": 1}}),
            "function",
        ));
        assert!(
            arguments.contains("messages[0].tool_calls[0].function.arguments"),
            "{arguments}"
        );
        let id = rejected(
            json!([{"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "{}"}}]}]),
        );
        assert!(id.contains("messages[0].tool_calls[0].id"), "{id}");
    }

    #[test]
    fn tools_pass_to_the_template_exactly_as_sent() {
        let sent = json!([{
            "type": "function",
            "function": {"name": "add", "parameters": {"z": 1, "a": 2}, "strict": true},
            "x-extra": true
        }]);
        let tools = parse_tools(Some(&sent)).expect("valid tools");
        assert_eq!(tools.len(), 1);
        assert_eq!(
            serde_json::to_string(&tools[0].0).expect("serialize"),
            serde_json::to_string(&sent[0]).expect("serialize"),
            "key order or content changed"
        );
        assert!(parse_tools(None).expect("absent").is_empty());
    }

    #[test]
    fn a_broken_tool_names_its_member() {
        let kind = parse_tools(Some(
            &json!([{"type": "plugin", "function": {"name": "f"}}]),
        ))
        .expect_err("kind");
        assert!(message_of(kind).contains("tools[0].type"));
        let name =
            parse_tools(Some(&json!([{"type": "function", "function": {}}]))).expect_err("name");
        assert!(message_of(name).contains("tools[0].function.name"));
        let shape = parse_tools(Some(&json!({"type": "function"}))).expect_err("shape");
        assert!(message_of(shape).contains("tools must be an array"));
    }

    #[test]
    fn a_tool_description_must_be_a_string() {
        let tool = |description: Value| json!([{"type": "function", "function": {"name": "f", "description": description}}]);
        assert!(parse_tools(Some(&tool(json!("adds")))).is_ok());
        let error = parse_tools(Some(&tool(json!([])))).expect_err("description");
        assert!(message_of(error).contains("tools[0].function.description"));
    }

    #[test]
    fn tool_parameters_must_be_an_object() {
        let tool = |parameters: Value| json!([{"type": "function", "function": {"name": "f", "parameters": parameters}}]);
        assert!(parse_tools(Some(&tool(json!({"type": "object"})))).is_ok());
        let error = parse_tools(Some(&tool(json!(5)))).expect_err("parameters");
        assert!(message_of(error).contains("tools[0].function.parameters"));
    }

    #[test]
    fn a_tool_strict_flag_must_be_a_boolean() {
        let tool = |strict: Value| json!([{"type": "function", "function": {"name": "f", "strict": strict}}]);
        assert!(parse_tools(Some(&tool(json!(false)))).is_ok());
        let error = parse_tools(Some(&tool(json!("yes")))).expect_err("strict");
        assert!(message_of(error).contains("tools[0].function.strict"));
    }

    #[test]
    fn tool_choice_none_disables_the_tools_and_other_valid_values_keep_them() {
        assert!(parse_tool_choice(Some(&json!("none"))).expect("none"));
        assert!(!parse_tool_choice(Some(&json!("auto"))).expect("auto"));
        assert!(!parse_tool_choice(Some(&json!("required"))).expect("required"));
        assert!(!parse_tool_choice(None).expect("absent"));
        let named = json!({"type": "function", "function": {"name": "add"}});
        assert!(!parse_tool_choice(Some(&named)).expect("named"));
    }

    #[test]
    fn an_invalid_tool_choice_is_rejected() {
        for bad in [
            json!("sometimes"),
            json!(3),
            json!({"type": "plugin", "function": {"name": "f"}}),
            json!({"type": "function"}),
            json!({"type": "function", "function": {}}),
        ] {
            let error = parse_tool_choice(Some(&bad)).expect_err("must be rejected");
            assert!(message_of(error).contains("tool_choice"), "{bad}");
        }
    }
}
