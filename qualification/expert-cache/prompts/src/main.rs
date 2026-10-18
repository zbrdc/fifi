//! Builds the T0 prompts file (S3 plan section 3.3) from `source.json` with the engine tokenizer.
//!
//! ```text
//! trace-prompts --checkpoint <checkpoint dir> --source source.json --out prompts.proposed.json
//! ```
//!
//! `raw` prompts encode their text without a template and without special tokens; a `raw` prompt
//! with `tokens` must encode to exactly those ids. `chat` prompts render `messages` and `tools`
//! through the pinned chat template with the generation prompt and encode the result
//! (`TokenizerBundle::encode_prompt`). `generate` is a token count, or `{"fill_to_position": P}`
//! for as many tokens as end the prompt at position `P` (the dense QSA limit).

use inference_engine::tokenizer::{
    PromptMessage, Role, Thinking, TokenizerBundle, TokenizerPin, ToolSpec,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("trace-prompts: {message}");
            ExitCode::FAILURE
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn digest(text: &str) -> Result<[u8; 32], String> {
    let bytes: Vec<u8> = (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(text.get(i..i + 2).unwrap_or("zz"), 16))
        .collect::<Result<_, _>>()
        .map_err(|_| format!("{text} is not hex"))?;
    bytes
        .try_into()
        .map_err(|_| format!("{text} is not 32 bytes"))
}

fn role(name: &str) -> Result<Role, String> {
    match name {
        "system" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        other => Err(format!("role {other}")),
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let (mut checkpoint, mut source, mut out) = (None, None, None);
    let mut iter = args.into_iter();
    while let Some(flag) = iter.next() {
        let value = PathBuf::from(iter.next().ok_or_else(|| format!("{flag} needs a value"))?);
        match flag.as_str() {
            "--checkpoint" => checkpoint = Some(value),
            "--source" => source = Some(value),
            "--out" => out = Some(value),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let checkpoint = checkpoint.ok_or("--checkpoint is required")?;
    let source = source.ok_or("--source is required")?;
    let out = out.ok_or("--out is required")?;
    let source_bytes = std::fs::read(&source).map_err(|e| format!("{}: {e}", source.display()))?;
    let spec: Value = serde_json::from_slice(&source_bytes).map_err(|e| format!("source: {e}"))?;
    let pin_value = &spec["tokenizer_pin"];
    let text = |key: &str| {
        pin_value[key]
            .as_str()
            .ok_or_else(|| format!("tokenizer_pin.{key}"))
    };
    let pin = TokenizerPin {
        tokenizer_json: digest(text("tokenizer_json")?)?,
        tokenizer_config_json: digest(text("tokenizer_config_json")?)?,
        chat_template_jinja: digest(text("chat_template_jinja")?)?,
        generation_config_json: digest(text("generation_config_json")?)?,
        vocab_size: pin_value["vocab_size"]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or("tokenizer_pin.vocab_size")?,
    };
    let bundle = TokenizerBundle::load(&checkpoint, &pin).map_err(|e| format!("{e:?}"))?;
    let mut prompts = Vec::new();
    for p in spec["prompts"].as_array().ok_or("source has no prompts")? {
        let id = p["id"].as_u64().ok_or("prompt id")?;
        let tokens = match p["kind"].as_str() {
            Some("raw") => {
                let text = p["text"].as_str().ok_or("raw prompt without text")?;
                let ids = bundle
                    .tokenizer()
                    .encode(text)
                    .map_err(|e| format!("{e:?}"))?;
                if let Some(want) = p.get("tokens") {
                    let want: Vec<u32> = serde_json::from_value(want.clone())
                        .map_err(|e| format!("prompt {id} tokens: {e}"))?;
                    if want != ids {
                        return Err(format!("prompt {id} encodes to {ids:?}, not {want:?}"));
                    }
                }
                ids
            }
            Some("chat") => {
                let mut messages = Vec::new();
                for m in p["messages"]
                    .as_array()
                    .ok_or("chat prompt without messages")?
                {
                    messages.push(PromptMessage {
                        role: role(m["role"].as_str().ok_or("message role")?)?,
                        content: m["content"].as_str().map(str::to_string),
                        name: None,
                        tool_call_id: None,
                        tool_calls: Vec::new(),
                    });
                }
                let tools: Vec<ToolSpec> = p["tools"]
                    .as_array()
                    .map(|list| list.iter().cloned().map(ToolSpec).collect())
                    .unwrap_or_default();
                let thinking = match p["thinking"].as_str() {
                    None | Some("default") => Thinking::Default,
                    Some("on") => Thinking::On,
                    Some("off") => Thinking::Off,
                    Some(other) => return Err(format!("thinking {other}")),
                };
                bundle
                    .encode_prompt(&messages, &tools, thinking)
                    .map_err(|e| format!("prompt {id}: {e:?}"))?
            }
            other => return Err(format!("prompt {id}: kind {other:?}")),
        };
        let generate = match &p["generate"] {
            Value::Number(n) => n.as_u64().ok_or("generate")?,
            Value::Object(o) => {
                let last = o["fill_to_position"].as_u64().ok_or("fill_to_position")?;
                // Steps are `tokens + generate - 1`, at positions 0 to `tokens + generate - 2`.
                (last + 2)
                    .checked_sub(tokens.len() as u64)
                    .filter(|&g| g >= 1)
                    .ok_or_else(|| format!("prompt {id}: {} tokens pass {last}", tokens.len()))?
            }
            _ => return Err(format!("prompt {id}: generate")),
        };
        println!(
            "prompt {id} {}: {} tokens, generate {generate}",
            p["name"].as_str().unwrap_or("-"),
            tokens.len()
        );
        prompts.push(json!({
            "id": id,
            "name": p["name"],
            "kind": p["kind"],
            "prompt_tokens": tokens.len(),
            "generate": generate,
            "tokens": tokens,
        }));
    }
    let head = json!({
        "schema": "s3-trace-prompts/1",
        "status": "proposed: the coordinator freezes this file and records its SHA-256",
        "source_sha256": hex(&Sha256::digest(&source_bytes)),
        "tokenizer": {
            "pin": pin_value,
            "rule": "raw: Tokenizer::encode(text) without special tokens; chat: TokenizerBundle::encode_prompt(messages, tools, thinking) with the pinned chat_template.jinja",
            "tool": "qualification/expert-cache/prompts (trace-prompts)",
        },
    });
    // The header is indented; each prompt stays on one line, so the long token lists stay short.
    let mut text = serde_json::to_string_pretty(&head).map_err(|e| e.to_string())?;
    let open = text.trim_end_matches('}').trim_end().len();
    text.truncate(open);
    text.push_str(",\n  \"prompts\": [\n");
    let lines: Vec<String> = prompts
        .iter()
        .map(|p| serde_json::to_string(p).map(|line| format!("    {line}")))
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    text.push_str(&lines.join(",\n"));
    text.push_str("\n  ]\n}\n");
    std::fs::write(&out, text).map_err(|e| format!("{}: {e}", out.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The byte-level BPE fixture of `engine/src/tokenizer/tests.rs`: "hello" is one token, id 8.
    const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":100,"content":" world","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":101,"content":"<|end|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"hello":8,"a":9,"b":10,"he":11,"hel":12,"hell":13},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;
    const TOKENIZER_CONFIG_JSON: &str =
        r#"{"tokenizer_class":"TestTokenizer","eos_token":"<|end|>"}"#;
    const GENERATION_CONFIG_JSON: &str = r#"{"eos_token_id":[11]}"#;
    /// Renders the first message and appends `hel` for the generation prompt: "hello" becomes
    /// "hellohel".
    const CHAT_TEMPLATE: &str =
        "{{ messages[0].content }}{% if add_generation_prompt %}hel{% endif %}";

    /// Writes the fixture checkpoint files and a source with `prompts`; returns the directory.
    fn fixture(name: &str, prompts: Value) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("trace-prompts-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let files = [
            ("tokenizer.json", TOKENIZER_JSON),
            ("tokenizer_config.json", TOKENIZER_CONFIG_JSON),
            ("chat_template.jinja", CHAT_TEMPLATE),
            ("generation_config.json", GENERATION_CONFIG_JSON),
        ];
        for (file, text) in files {
            std::fs::write(dir.join(file), text).unwrap();
        }
        let pin = |text: &str| hex(&Sha256::digest(text.as_bytes()));
        let source = json!({
            "tokenizer_pin": {
                "tokenizer_json": pin(TOKENIZER_JSON),
                "tokenizer_config_json": pin(TOKENIZER_CONFIG_JSON),
                "chat_template_jinja": pin(CHAT_TEMPLATE),
                "generation_config_json": pin(GENERATION_CONFIG_JSON),
                "vocab_size": 16,
            },
            "prompts": prompts,
        });
        std::fs::write(dir.join("source.json"), source.to_string()).unwrap();
        dir
    }

    fn args(dir: &std::path::Path) -> Vec<String> {
        let path = |p: PathBuf| p.to_str().unwrap().to_string();
        vec![
            "--checkpoint".into(),
            path(dir.to_path_buf()),
            "--source".into(),
            path(dir.join("source.json")),
            "--out".into(),
            path(dir.join("prompts.json")),
        ]
    }

    #[test]
    fn prompts_get_their_tokens_and_generation_counts() {
        let dir = fixture(
            "ok",
            json!([
                {"id": 0, "name": "raw", "kind": "raw", "text": "hello", "tokens": [8], "generate": 3},
                {"id": 1, "name": "chat", "kind": "chat", "thinking": "off",
                 "messages": [{"role": "user", "content": "hello"}],
                 "generate": {"fill_to_position": 5}},
            ]),
        );
        assert_eq!(run(args(&dir)), Ok(()));
        let out: Value =
            serde_json::from_slice(&std::fs::read(dir.join("prompts.json")).unwrap()).unwrap();
        // The chat prompt renders "hellohel": tokens 8 and 12. Two tokens end at position 5
        // after 5 + 2 - 2 = 5 generated tokens.
        assert_eq!(
            out["prompts"],
            json!([
                {"id": 0, "name": "raw", "kind": "raw", "prompt_tokens": 1, "generate": 3, "tokens": [8]},
                {"id": 1, "name": "chat", "kind": "chat", "prompt_tokens": 2, "generate": 5, "tokens": [8, 12]},
            ])
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_raw_prompt_must_encode_to_its_listed_tokens() {
        let dir = fixture(
            "mismatch",
            json!([{"id": 0, "name": "raw", "kind": "raw", "text": "hello", "tokens": [9], "generate": 1}]),
        );
        let result = run(args(&dir));
        assert!(
            result
                .as_ref()
                .is_err_and(|m| m.contains("encodes to [8], not [9]")),
            "{result:?}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
