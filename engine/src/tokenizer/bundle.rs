use super::{ChatTemplate, PromptMessage, Role, Thinking, Tokenizer, ToolSpec};
use crate::{EngError, checkpoint_json::UniqueValue};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

#[derive(Clone, Copy)]
pub struct TokenizerIdentity {
    pub tokenizer_json: [u8; 32],
    pub tokenizer_config_json: [u8; 32],
    pub chat_template_jinja: [u8; 32],
    pub generation_config_json: [u8; 32],
}

#[derive(Clone, Copy)]
pub struct TokenizerPin {
    pub tokenizer_json: [u8; 32],
    pub tokenizer_config_json: [u8; 32],
    pub chat_template_jinja: [u8; 32],
    pub generation_config_json: [u8; 32],
    pub vocab_size: u32,
}

/// The message-boundary token counts of one encoded prompt (decision 0036, addenda 1 and 2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PromptBoundaries {
    /// The number of prompt tokens that end at the last message boundary, when the prompt ids
    /// extend them: the prompt without its generation tail. `None` turns prefix reuse off.
    pub tip: Option<usize>,
    /// The number of prompt tokens that end at the first system message, when the prompt ids
    /// extend them and a tip follows them. `None` when the first message is not a system
    /// message, the render of the system block fails or does not end in the text of an empty
    /// user turn, there is no tip, or the two boundaries meet.
    pub root: Option<usize>,
}

pub struct TokenizerBundle {
    identity: TokenizerIdentity,
    tokenizer: Tokenizer,
    eos_token_ids: Vec<u32>,
    vocab_size: u32,
    template: ChatTemplate,
}

impl TokenizerBundle {
    pub fn load(dir: &Path, pin: &TokenizerPin) -> Result<Self, EngError> {
        let tokenizer_bytes = read_pinned(dir, "tokenizer.json", pin.tokenizer_json)?;
        let config_bytes = read_pinned(dir, "tokenizer_config.json", pin.tokenizer_config_json)?;
        let template_bytes = read_pinned(dir, "chat_template.jinja", pin.chat_template_jinja)?;
        let generation_bytes =
            read_pinned(dir, "generation_config.json", pin.generation_config_json)?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)?;
        let vocab_size = checked_vocab_size(&tokenizer, pin.vocab_size)?;
        let config = parse_json(&config_bytes, "tokenizer_config.json")?;
        let generation = parse_json(&generation_bytes, "generation_config.json")?;
        let eos_token_ids = eos_lookup(&config, &generation, &tokenizer)?;
        let template = parse_template(&template_bytes)?;
        Ok(Self {
            identity: TokenizerIdentity {
                tokenizer_json: pin.tokenizer_json,
                tokenizer_config_json: pin.tokenizer_config_json,
                chat_template_jinja: pin.chat_template_jinja,
                generation_config_json: pin.generation_config_json,
            },
            tokenizer,
            eos_token_ids,
            vocab_size,
            template,
        })
    }

    pub fn identity(&self) -> &TokenizerIdentity {
        &self.identity
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The id of `eos_token` in `tokenizer_config.json`. It is the first id of
    /// `eos_token_ids`.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_ids[0]
    }

    /// Every id that ends generation, without duplicates. The `eos_token` id
    /// comes first. The ids of `eos_token_id` in `generation_config.json`
    /// follow in file order. The reference loader builds the same set.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub fn encode_prompt(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
    ) -> Result<Vec<u32>, EngError> {
        let rendered = self.template.render(msgs, tools, thinking)?;
        if rendered.is_empty() {
            return Err(EngError::BadRequest("rendered prompt is empty".to_string()));
        }
        self.tokenizer.encode(&rendered)
    }

    /// Encodes the prompt like `encode_prompt` and also returns its message boundaries
    /// (decision 0036, addenda 1 and 2): the `tip` is the prompt length at the last message
    /// boundary, the prompt without its generation tail, and the `root` is the prompt length
    /// at the end of the first system message. A boundary is `None` unless the ids of its
    /// render are the first ids of the prompt. A set boundary always has `0 < n < ids.len()`,
    /// and the root is always shorter than the tip.
    pub fn encode_prompt_with_boundary(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
    ) -> Result<(Vec<u32>, PromptBoundaries), EngError> {
        let ids = self.encode_prompt(msgs, tools, thinking)?;
        let head = self
            .template
            .render_without_generation_prompt(msgs, tools, thinking)?;
        let head_ids = self.tokenizer.encode(&head)?;
        let tip = Some(head_ids.len())
            .filter(|&len| len > 0 && len < ids.len() && ids.starts_with(&head_ids));
        let root = self.root_boundary(msgs, tools, thinking, &ids, tip);
        Ok((ids, PromptBoundaries { tip, root }))
    }

    /// The token count at the end of the first system message. The pinned Qwen template raises
    /// `No user query found in messages.` for every render without a user turn, so the first
    /// message cannot be rendered alone. This renders the first message plus an empty user turn
    /// (same tools and thinking setting, no generation prompt) and cuts the text of an empty user
    /// turn, which the template writes last, from the end. The system block (the tools block
    /// included) comes first in the pinned template and does not depend on later messages, so
    /// the rest is the text of the first system message. The ids of that text must be the first
    /// ids of the prompt, shorter than the tip. When the tip is `None`, the root is dropped too.
    /// Every error of the root render gives `None`: the root only helps reuse, and the full
    /// prompt rendered already.
    fn root_boundary(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
        ids: &[u32],
        tip: Option<usize>,
    ) -> Option<usize> {
        let tip = tip?;
        let first = msgs.first().filter(|m| matches!(m.role, Role::System))?;
        let empty_user = PromptMessage {
            role: Role::User,
            content: Some(String::new()),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        };
        let stub = self
            .template
            .render_without_generation_prompt(std::slice::from_ref(&empty_user), &[], thinking)
            .ok()?;
        let render = self
            .template
            .render_without_generation_prompt(&[first.clone(), empty_user], tools, thinking)
            .ok()?;
        let root_text = render.strip_suffix(stub.as_str())?;
        let root_ids = self.tokenizer.encode(root_text).ok()?;
        Some(root_ids.len())
            .filter(|&len| len > 0 && len < ids.len() && len < tip && ids.starts_with(&root_ids))
    }
}

fn read_pinned(dir: &Path, name: &str, expected: [u8; 32]) -> Result<Vec<u8>, EngError> {
    let bytes = fs::read(dir.join(name)).map_err(|error| {
        if error.kind() == ErrorKind::NotFound {
            EngError::Checkpoint(format!("{name} is missing from the checkpoint directory"))
        } else {
            EngError::Checkpoint(format!("{name} could not be read: {error}"))
        }
    })?;
    if digest(&bytes) != expected {
        return Err(EngError::Checkpoint(format!(
            "{name} hash does not match the pinned revision"
        )));
    }
    Ok(bytes)
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

fn parse_json(bytes: &[u8], name: &str) -> Result<serde_json::Value, EngError> {
    serde_json::from_slice::<UniqueValue>(bytes)
        .map(|value| value.0)
        .map_err(|error| EngError::Checkpoint(format!("{name} parse failed: {error}")))
}

fn checked_vocab_size(tokenizer: &Tokenizer, expected: u32) -> Result<u32, EngError> {
    let vocab_size = tokenizer.vocab_size();
    if vocab_size != expected {
        return Err(EngError::Checkpoint(format!(
            "tokenizer vocab size {vocab_size} does not match the pinned vocab size {expected}"
        )));
    }
    Ok(vocab_size)
}

fn eos_lookup(
    config: &serde_json::Value,
    generation: &serde_json::Value,
    tokenizer: &Tokenizer,
) -> Result<Vec<u32>, EngError> {
    let name = config
        .get("eos_token")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            EngError::Checkpoint("tokenizer_config.json does not define eos_token".to_string())
        })?;
    let primary = tokenizer.token_id(name).ok_or_else(|| {
        EngError::Checkpoint(format!(
            "eos_token {name} is not in the tokenizer vocabulary"
        ))
    })?;
    let mut ids = vec![primary];
    for id in generation_eos_ids(generation, tokenizer.vocab_size())? {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    Ok(ids)
}

// `eos_token_id` in generation_config.json is one integer or a list of
// integers. An absent key adds no id.
fn generation_eos_ids(
    generation: &serde_json::Value,
    vocab_size: u32,
) -> Result<Vec<u32>, EngError> {
    let Some(value) = generation.get("eos_token_id") else {
        return Ok(Vec::new());
    };
    let items = match value {
        serde_json::Value::Array(items) => items.as_slice(),
        single => std::slice::from_ref(single),
    };
    items
        .iter()
        .map(|item| {
            let id = item
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| {
                    EngError::Checkpoint(format!(
                        "generation_config.json eos_token_id must hold non-negative integers, found {item}"
                    ))
                })?;
            if id >= vocab_size {
                return Err(EngError::Checkpoint(format!(
                    "generation_config.json eos_token_id {id} is outside the tokenizer vocabulary of {vocab_size} ids"
                )));
            }
            Ok(id)
        })
        .collect()
}

fn parse_template(bytes: &[u8]) -> Result<ChatTemplate, EngError> {
    let source = std::str::from_utf8(bytes)
        .map_err(|_| EngError::Checkpoint("chat_template.jinja is not UTF-8".to_string()))?;
    ChatTemplate::from_str(source)
}
