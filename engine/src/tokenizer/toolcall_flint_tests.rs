use super::*;
use serde_json::json;

fn schemas() -> ToolSchemas {
    ToolSchemas::from_tools(&[ToolSpec(json!({"type":"function","function":{
        "name":"run", "parameters":{"type":"object","properties":{
            "code":{"type":"string"}, "literal":{"type":"string"}, "n":{"type":"integer"}
        }}
    }}))])
}

fn parse(parts: &[&str], grammar: ToolGrammar) -> ParsedOutput {
    let mut parser = ToolCallParser::with_grammar(Start::Content, schemas(), "call_", grammar);
    let mut events = Vec::new();
    for part in parts {
        events.extend(parser.push(part));
    }
    events.extend(parser.finish());
    ParsedOutput::from_events(events)
}

#[test]
fn flint_all_marker_and_argument_splits_share_the_same_framing() {
    let text = concat!(
        "<think>Check λ.</think>Running.",
        "<tool_call>run<arg_key>code</arg_key><arg_value>  print(\"🌍\")\n </arg_value>",
        "<arg_key>literal</arg_key><arg_value>123</arg_value>",
        "<arg_key>n</arg_key><arg_value>2</arg_value>",
        "<arg_key>options</arg_key><arg_value>{\"strict\":true,\"x\":[1,null]}</arg_value></tool_call>",
        "<tool_call>done</tool_call>Done."
    );
    let expected = parse(&[text], ToolGrammar::Flint);
    assert_eq!(expected.reasoning, "Check λ.");
    assert_eq!(expected.content, "Running.Done.");
    assert!(expected.malformed.is_empty());
    assert_eq!(expected.tool_calls.len(), 2);
    assert_eq!(expected.tool_calls[0].id, "call_0");
    assert_eq!(expected.tool_calls[0].name, "run");
    assert_eq!(
        serde_json::from_str::<Value>(&expected.tool_calls[0].arguments).unwrap(),
        json!({
            "code":"  print(\"🌍\")\n ", "literal":"123", "n":2,
            "options":{"strict":true,"x":[1,null]}
        })
    );
    assert_eq!(expected.tool_calls[1].id, "call_1");
    assert_eq!(expected.tool_calls[1].arguments, "{}");
    for split in 0..=text.len() {
        if text.is_char_boundary(split) {
            assert_eq!(
                parse(&[&text[..split], &text[split..]], ToolGrammar::Flint),
                expected,
                "split {split}"
            );
        }
    }
    let pieces: Vec<_> = text
        .char_indices()
        .map(|(i, c)| &text[i..i + c.len_utf8()])
        .collect();
    assert_eq!(parse(&pieces, ToolGrammar::Flint), expected);
}

#[test]
fn flint_rejects_malformed_arguments_without_reinterpreting_qwen() {
    let bodies = [
        ("", ToolCallError::InvalidName),
        ("run<arg_key>", ToolCallError::UnclosedTag),
        (
            "run<arg_key></arg_key><arg_value>x</arg_value>",
            ToolCallError::InvalidName,
        ),
        ("run<arg_key>x</arg_key>", ToolCallError::UnexpectedText),
        (
            "run<arg_key>x</arg_key><arg_value>1",
            ToolCallError::UnclosedParameter,
        ),
        (
            "run<arg_key>x</arg_key><arg_value>1</arg_value>junk",
            ToolCallError::UnexpectedText,
        ),
        (
            "run<arg_key>x</arg_key><arg_value>1</arg_value><arg_key>x</arg_key><arg_value>2</arg_value>",
            ToolCallError::DuplicateParameter("x".into()),
        ),
        ("<function=run></function>", ToolCallError::InvalidName),
    ];
    for (body, error) in bodies {
        let text = format!("<tool_call>{body}</tool_call>");
        let result = parse(&[&text], ToolGrammar::Flint);
        assert!(result.tool_calls.is_empty(), "{body}");
        assert!(
            result.content.is_empty(),
            "malformed calls must not leak into content framing"
        );
        assert_eq!(
            result.malformed,
            [MalformedCall {
                error,
                raw: body.into()
            }]
        );
    }
    let flint = "<tool_call>run<arg_key>n</arg_key><arg_value>2</arg_value></tool_call>";
    let legacy = parse_complete(flint, Start::Content, schemas(), "call_");
    assert!(legacy.tool_calls.is_empty());
    assert_eq!(legacy.malformed[0].error, ToolCallError::MissingFunction);
    let qwen = parse_complete(
        "<tool_call><function=run></function></tool_call>",
        Start::Content,
        schemas(),
        "call_",
    );
    assert_eq!(qwen.tool_calls[0].name, "run");
}

#[test]
fn flint_truncation_and_reasoning_start_keep_existing_finish_rules() {
    let raw = "run<arg_key>code</arg_key><arg_value>λ";
    let output = parse(&["ok<tool_", "call>", raw], ToolGrammar::Flint);
    assert_eq!(output.content, "ok");
    assert!(output.tool_calls.is_empty());
    assert_eq!(
        output.malformed,
        [MalformedCall {
            error: ToolCallError::Unterminated,
            raw: raw.into()
        }]
    );
    let mut parser =
        ToolCallParser::with_grammar(Start::Reasoning, schemas(), "r_", ToolGrammar::Flint);
    let mut events = parser.push("plan</thi");
    events.extend(parser.push("nk><tool_call>run</tool_call>"));
    events.extend(parser.finish());
    let output = ParsedOutput::from_events(events);
    assert_eq!(output.reasoning, "plan");
    assert_eq!(output.tool_calls[0].name, "run");
    assert!(output.content.is_empty());
}
