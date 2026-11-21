use super::{ChatTemplate, TokenizerBundle, TokenizerPin};
use crate::EngError;
use std::path::Path;

// thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7
const MINI_PIN: TokenizerPin = TokenizerPin {
    tokenizer_json: [
        0x7f, 0x33, 0x27, 0x27, 0xf9, 0xa3, 0xf0, 0x93, 0xdd, 0x45, 0x2a, 0x39, 0xcc, 0x97, 0x8e,
        0x67, 0x57, 0x62, 0x7e, 0x5d, 0xad, 0x1b, 0x06, 0x73, 0xaa, 0x4e, 0x75, 0x17, 0x4f, 0x23,
        0x6d, 0xd6,
    ],
    tokenizer_config_json: [
        0xbf, 0x09, 0x34, 0x2d, 0x28, 0x51, 0x39, 0xfd, 0x37, 0xce, 0x10, 0xc7, 0x08, 0x78, 0xaf,
        0x1d, 0x52, 0xcb, 0x39, 0x0e, 0x99, 0x01, 0xe1, 0x02, 0x48, 0x65, 0x42, 0x93, 0xa8, 0x0e,
        0xe6, 0x36,
    ],
    chat_template_jinja: [
        0x53, 0xa9, 0x03, 0x5f, 0x32, 0x55, 0x94, 0x57, 0x04, 0xb5, 0x85, 0x76, 0x57, 0xcf, 0xb0,
        0x0c, 0x11, 0x65, 0x09, 0x0c, 0x06, 0x41, 0xc4, 0xf7, 0x82, 0xa0, 0xcb, 0xf2, 0x68, 0xcc,
        0x16, 0xdc,
    ],
    generation_config_json: [
        0xa0, 0x87, 0x37, 0xc0, 0x7a, 0x65, 0xa8, 0xfd, 0xf6, 0x9f, 0x7c, 0xf0, 0x86, 0x57, 0x62,
        0x10, 0xe5, 0xaa, 0xd4, 0xb2, 0x41, 0x43, 0x70, 0x90, 0xb2, 0x6d, 0x7b, 0x0c, 0x4f, 0x79,
        0x05, 0xf7,
    ],
    vocab_size: 200_058,
};

pub(crate) fn load(dir: &Path) -> Result<TokenizerBundle, EngError> {
    let bundle = TokenizerBundle::load_with_template(dir, &MINI_PIN, text_template)?;
    if !bundle.load_notes().is_empty() {
        return Err(EngError::Checkpoint(
            "Inkling 2 Mini CPU requires the tokenizer files from the pinned Mini release".into(),
        ));
    }
    Ok(bundle)
}

fn text_template(source: &str) -> Result<ChatTemplate, EngError> {
    // The Transformers generation block annotates assistant spans without changing their text.
    let source = source
        .replace("{%- generation -%}", "{%- if true -%}")
        .replace("{%- endgeneration -%}", "{%- endif -%}");
    const TEXT_ONLY: &str = r#"
{%- if enable_thinking is defined and enable_thinking -%}
    {{- raise_exception("Inkling CPU chat supports thinking off") -}}
{%- endif -%}
{%- if tools -%}
    {{- raise_exception("Inkling CPU chat supports text without tools") -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if message.role == "tool" or message.tool_calls -%}
        {{- raise_exception("Inkling CPU chat supports text without tool history") -}}
    {%- endif -%}
{%- endfor -%}
{%- set reasoning_effort = "none" -%}
"#;
    // Zero effort still permits a reasoning span. Open the assistant text span explicitly
    // because this backend returns plain chat text.
    const TEXT_START: &str =
        "{%- if add_generation_prompt -%}<|message_model|><|content_text|>{%- endif -%}";
    ChatTemplate::from_str(&format!("{TEXT_ONLY}{source}{TEXT_START}"))
}

#[cfg(test)]
mod tests {
    use super::text_template;
    use crate::{
        EngError,
        tokenizer::{PromptMessage, Role, Thinking, ToolSpec},
    };

    fn message(role: Role) -> PromptMessage {
        PromptMessage {
            role,
            content: Some("hello".into()),
            name: None,
            reasoning_content: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    #[test]
    fn strips_generation_annotations_and_sets_zero_effort() {
        let template = text_template(
            "{{ reasoning_effort }}:{%- generation -%} assistant {%- endgeneration -%}",
        )
        .unwrap();
        let rendered = template
            .render(&[message(Role::Assistant)], &[], Thinking::Default)
            .unwrap();
        assert_eq!(rendered, "none:assistant<|message_model|><|content_text|>");
        assert_eq!(
            template
                .render_without_generation_prompt(
                    &[message(Role::Assistant)],
                    &[],
                    Thinking::Default
                )
                .unwrap(),
            "none:assistant"
        );
    }

    #[test]
    fn only_plain_text_chat_is_admitted() {
        let template = text_template("text").unwrap();
        let user = [message(Role::User)];
        let tool = [message(Role::Tool)];
        let tools = [ToolSpec(serde_json::json!({"function": {"name": "f"}}))];
        for (messages, declarations, thinking) in [
            (&user[..], &tools[..], Thinking::Default),
            (&tool[..], &[][..], Thinking::Default),
            (&user[..], &[][..], Thinking::On),
        ] {
            assert!(matches!(
                template.render(messages, declarations, thinking),
                Err(EngError::BadRequest(_))
            ));
        }
        assert_eq!(
            template.render(&user, &[], Thinking::Off).unwrap(),
            "text<|message_model|><|content_text|>"
        );
    }
}
