use super::{PromptMessage, Thinking, ToolSpec, pyjson, pystr};
use crate::EngError;
use minijinja::{Error, ErrorKind, Value};

pub struct ChatTemplate {
    env: minijinja::Environment<'static>,
}

impl ChatTemplate {
    pub(crate) fn from_str(source: &str) -> Result<Self, EngError> {
        let mut env = minijinja::Environment::new();
        // The reference renders with trim_blocks and lstrip_blocks enabled.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.add_global("raise_exception", Value::from_function(raise_exception));
        env.add_filter("tojson", pyjson::tojson_filter);
        env.add_filter("trim", pystr::trim_filter);
        env.set_unknown_method_callback(unknown_method);
        // The engine loads one template per process. The owned source keeps the
        // environment free of lifetimes without leaking the bytes.
        env.add_template_owned("chat", normalize_newlines(source))
            .map_err(|error| {
                EngError::Checkpoint(format!("chat_template.jinja syntax error: {error}"))
            })?;
        Ok(Self { env })
    }

    pub fn render(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
    ) -> Result<String, EngError> {
        self.render_with(msgs, tools, thinking, true)
    }

    /// Renders the messages without the generation prompt. For a template that appends its
    /// generation prompt, this is the text up to the last message boundary.
    pub fn render_without_generation_prompt(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
    ) -> Result<String, EngError> {
        self.render_with(msgs, tools, thinking, false)
    }

    fn render_with(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
        generation_prompt: bool,
    ) -> Result<String, EngError> {
        if msgs.is_empty() {
            return Err(EngError::BadRequest("message list is empty".to_string()));
        }
        let context = build_context(msgs, tools, thinking, generation_prompt)?;
        let template = self.env.get_template("chat").map_err(render_defect)?;
        template.render(context).map_err(render_error)
    }
}

// Jinja turns every newline sequence in template text into "\n".
fn normalize_newlines(source: &str) -> String {
    source.replace("\r\n", "\n").replace('\r', "\n")
}

// Marks an error that `raise_exception` produced. The template author chose to
// reject this request, so the cause is the request. Every other render error is
// a template or engine defect.
#[derive(Debug)]
struct TemplateRaised;

impl std::fmt::Display for TemplateRaised {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("raised by raise_exception")
    }
}

impl std::error::Error for TemplateRaised {}

fn raise_exception(message: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message).with_source(TemplateRaised))
}

fn is_template_raise(error: &Error) -> bool {
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        if cause.is::<TemplateRaised>() {
            return true;
        }
        source = cause.source();
    }
    false
}

fn unknown_method(
    _state: &minijinja::State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    let Some(text) = value.as_str() else {
        return Err(unknown_name(method));
    };
    pystr::call(text, method, args).unwrap_or_else(|| Err(unknown_name(method)))
}

fn unknown_name(method: &str) -> Error {
    Error::new(
        ErrorKind::UnknownMethod,
        format!("unknown template method {method}"),
    )
}

fn build_context(
    msgs: &[PromptMessage],
    tools: &[ToolSpec],
    thinking: Thinking,
    generation_prompt: bool,
) -> Result<Value, EngError> {
    let mut context = serde_json::Map::new();
    context.insert("messages".to_string(), message_context(msgs)?);
    context.insert("tools".to_string(), to_value(tools, "tools")?);
    context.insert(
        "add_generation_prompt".to_string(),
        serde_json::json!(generation_prompt),
    );
    if !matches!(thinking, Thinking::Default) {
        context.insert(
            "enable_thinking".to_string(),
            serde_json::json!(matches!(thinking, Thinking::On)),
        );
    }
    Ok(Value::from_serialize(serde_json::Value::Object(context)))
}

fn message_context(msgs: &[PromptMessage]) -> Result<serde_json::Value, EngError> {
    let mut messages = to_value(msgs, "messages")?;
    let rows = messages
        .as_array_mut()
        .ok_or_else(|| EngError::Checkpoint("messages template context failed".to_string()))?;
    for row in rows {
        let Some(calls) = row
            .get_mut("tool_calls")
            .and_then(serde_json::Value::as_array_mut)
        else {
            continue;
        };
        for call in calls {
            let text = call
                .get("arguments")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    EngError::BadRequest("tool call arguments must be JSON text".to_string())
                })?;
            let parsed: serde_json::Value = serde_json::from_str(text).map_err(|error| {
                EngError::BadRequest(format!("tool call arguments failed: {error}"))
            })?;
            if !parsed.is_object() {
                return Err(EngError::BadRequest(
                    "tool call arguments must be a JSON object".to_string(),
                ));
            }
            call["arguments"] = parsed;
        }
    }
    Ok(messages)
}

fn to_value<T: serde::Serialize + ?Sized>(
    value: &T,
    what: &str,
) -> Result<serde_json::Value, EngError> {
    serde_json::to_value(value)
        .map_err(|error| EngError::Checkpoint(format!("{what} template context failed: {error}")))
}

fn render_defect(error: Error) -> EngError {
    EngError::Checkpoint(format!("chat template failed: {error}"))
}

fn render_error(error: Error) -> EngError {
    if is_template_raise(&error) {
        EngError::BadRequest(format!("chat template rejected the request: {error}"))
    } else {
        render_defect(error)
    }
}
