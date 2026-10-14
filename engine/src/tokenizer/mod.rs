//! Tokenizer and chat-template ownership for the engine.
//!
//! Interface and rules: the tokenizer design.
//! Test seams: `tokenizer/seams.md`, in the ledger.
//! Dependency decision: decision 0025.

mod bundle;
mod decode;
mod encode;
mod output;
mod pyjson;
mod pystr;
mod stop;
mod template;
pub mod toolcall;

pub use bundle::{PromptBoundaries, TokenizerBundle, TokenizerIdentity, TokenizerPin};
pub use decode::IncrementalDecoder;
pub use encode::Tokenizer;
pub use output::{DecodeBatch, DecodeFinish, OutputAssembler, OutputDelta};
pub use stop::{StopHit, StopMatcher};
pub use template::ChatTemplate;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Copy, Debug)]
pub enum Thinking {
    Default,
    On,
    Off,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ToolCallSpec {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PromptMessage {
    pub role: Role,
    pub content: Option<String>,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_calls: Vec<ToolCallSpec>,
}

/// One tool definition, exactly as the client sent it.
///
/// The template receives this JSON value unchanged: key order, nesting, and the
/// OpenAI `{"type": "function", "function": {...}}` wrapper stay as sent.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(transparent)]
pub struct ToolSpec(pub serde_json::Value);
