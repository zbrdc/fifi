//! Tokenizer and chat-template ownership for the engine.
//!
//! Interface and rules: the tokenizer design.
//! Test seams: `tokenizer/seams.md`, in the ledger.
//! Dependency decision: decision 0025.

mod bundle;
mod decode;
mod encode;
pub mod nemotron;
mod output;
mod pyjson;
mod pystr;
mod stop;
mod template;
pub mod toolcall;

pub use bundle::{
    PromptBoundaries, TokenizerBundle, TokenizerIdentity, TokenizerPin, WarmTurnCache,
};
pub use decode::IncrementalDecoder;
pub use encode::Tokenizer;
pub use output::{DecodeBatch, DecodeFinish, OutputAssembler, OutputDelta};
pub use stop::{StopHit, StopMatcher};
pub use template::ChatTemplate;

#[cfg(test)]
mod boundary_derive_tests;
#[cfg(test)]
pub(crate) mod flint_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod warm_turn_tests;

/// The fail-safe default-on env switch shared by the tokenizer replay flags
/// and the Flint family switches (`FIFI_FLINT_MLX_ASYNC_CHAIN`,
/// `FIFI_FLINT_MLX_COALESCE_RESIDUAL`, `FIFI_FLINT_MLX_DECODE_ATTN`,
/// `FIFI_FLINT_MLX_RING_VIEW`, `FIFI_FLINT_CUDA_ALLOC_CACHE`): an absent
/// variable keeps the new default (on); once set, only the explicit enable
/// words ("1"/"true"/"on", trimmed, case-folded) keep the feature on, and
/// anything else — disable words, case games, whitespace, CRLF, typos, a
/// value that is not UTF-8 — rolls back. A rollback can never silently
/// fail.
pub(crate) fn env_allows_default_on(name: &str) -> bool {
    env_result_allows_default_on(std::env::var(name))
}

/// The pure `Result`-shaped core of [`env_allows_default_on`]: absent keeps
/// the default on; non-UTF-8 and any set value resolve through
/// [`env_allows_value`]. Shared with switch helpers that receive an
/// already-read env result (e.g. the ring-view flag) so the polarity match
/// exists in exactly one place.
pub(crate) fn env_result_allows_default_on(value: Result<String, std::env::VarError>) -> bool {
    match value {
        // Only a genuinely absent variable keeps the new default; a set
        // value that is not UTF-8 is a mis-set rollback and disables.
        Err(std::env::VarError::NotPresent) => true,
        Err(std::env::VarError::NotUnicode(_)) => false,
        Ok(value) => env_allows_value(Some(&value)),
    }
}

/// The pure polarity half of [`env_allows_default_on`], testable without
/// touching process-global environment.
pub(crate) fn env_allows_value(value: Option<&str>) -> bool {
    match value {
        None => true,
        Some(text) => matches!(
            text.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on"
        ),
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    Developer,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
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
