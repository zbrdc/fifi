//! The pinned Humanizer completion prompt of the Gemma 4 Humanizer model.
//!
//! The model README defines the prompt as the instruction, a blank line, the draft
//! without surrounding whitespace, and a fixed separator. The published template is
//! a plain completion template: it carries no chat markers, and the BOS token is
//! the tokenizer's beginning-of-sequence id, written as the literal text the pinned
//! tokenizer maps to id 2.
//!
//! The instruction and separator here are the pinned strings of
//! testdata/gemma4/prompt-cases.json, whose prompt hashes are fixed.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged until the server wires the prompt")
)]

use crate::EngError;
use crate::tokenizer::TokenizerBundle;

/// The pinned rewriting instruction.
pub(crate) const INSTRUCTION: &str = "Rewrite the text below so it reads like a person wrote it, not a language model.\n\nReorganize it as you see fit. Vary sentence length on purpose. Cut hedging,\nthroat-clearing, and any sentence that only announces what comes next.\nPrefer the concrete word over the abstract one. It is fine to sound uneven.\n\nEvery fact, number, unit, date, name and quotation must survive unchanged.";

/// The pinned separator between the draft and the continuation point.
pub(crate) const SEPARATOR: &str = "\n\n### Rewritten:\n\n";

/// The literal text of the beginning-of-sequence token in the pinned tokenizer.
pub(crate) const BOS_TEXT: &str = "<bos>";

/// Renders the pinned prompt for one draft: instruction, blank line, the stripped
/// draft, then the separator. History and system messages do not enter this prompt.
pub(crate) fn render(draft: &str) -> String {
    format!("{INSTRUCTION}\n\n{}{SEPARATOR}", draft.trim())
}

/// Encodes one draft into the prompt tokens, beginning with the BOS id.
///
/// The pressed template is unchanged; the BOS id comes from the literal BOS text, so
/// the encoder stays unchanged and no special-token switch is enabled.
pub(crate) fn encode(bundle: &TokenizerBundle, draft: &str) -> Result<Vec<u32>, EngError> {
    let text = format!("{BOS_TEXT}{}", render(draft));
    bundle.tokenizer().encode(&text)
}

#[cfg(test)]
mod tests {
    use super::{BOS_TEXT, INSTRUCTION, SEPARATOR, encode, render};
    use crate::EngError;
    use crate::tokenizer::{Tokenizer, TokenizerBundle, TokenizerPin};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::fs;

    const CASES: &[u8] = include_bytes!("../../../testdata/gemma4/prompt-cases.json");

    /// The id of `<bos>` in the pinned tokenizer.
    const BOS_ID: u32 = 2;
    /// The first id of the character vocabulary: ids 0 to 2 are `<pad>`, `<eos>` and `<bos>`.
    const FIRST_CHAR_ID: u32 = 3;
    const TOKENIZER_CONFIG_JSON: &str = r#"{"eos_token":"<eos>"}"#;
    const GENERATION_CONFIG_JSON: &str = r#"{"eos_token_id":1}"#;
    const CHAT_TEMPLATE: &str = "{{ messages[0].content }}";

    fn cases() -> serde_json::Value {
        serde_json::from_slice(CASES).expect("prompt cases")
    }

    fn special_token(id: u32, content: &str) -> serde_json::Value {
        serde_json::json!({
            "id": id, "content": content, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true
        })
    }

    /// The ids of a character-level tokenizer: one id per distinct character of `texts`, in
    /// character order from `FIRST_CHAR_ID`.
    fn char_ids(texts: &[&str]) -> BTreeMap<char, u32> {
        let mut chars: Vec<char> = texts.iter().flat_map(|text| text.chars()).collect();
        chars.sort_unstable();
        chars.dedup();
        (FIRST_CHAR_ID..)
            .zip(chars)
            .map(|(id, c)| (c, id))
            .collect()
    }

    /// A character-level BPE tokenizer.json shaped like the pinned one: `<bos>` is a special
    /// added token and a vocabulary entry with id 2 (the tokenizer keeps the vocabulary id for
    /// an added token it already holds), and a post-processor would add a `<bos>` of its own
    /// when special tokens are switched on. `unk_token` names the token for characters that are
    /// not in `chars`.
    fn tokenizer_json(chars: &BTreeMap<char, u32>, unk_token: Option<&str>) -> String {
        let specials = [("<pad>", 0), ("<eos>", 1), ("<bos>", BOS_ID)];
        let vocab: serde_json::Map<String, serde_json::Value> = specials
            .into_iter()
            .map(|(content, id)| (content.to_string(), serde_json::json!(id)))
            .chain(
                chars
                    .iter()
                    .map(|(c, id)| (c.to_string(), serde_json::json!(id))),
            )
            .collect();
        let single = serde_json::json!([
            {"SpecialToken": {"id": "<bos>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}}
        ]);
        serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [
                special_token(0, "<pad>"), special_token(1, "<eos>"), special_token(BOS_ID, "<bos>")
            ],
            "normalizer": null, "pre_tokenizer": null,
            "post_processor": {
                "type": "TemplateProcessing", "single": single, "pair": single,
                "special_tokens": {"<bos>": {"id": "<bos>", "ids": [BOS_ID], "tokens": ["<bos>"]}}
            },
            "decoder": null,
            "model": {
                "type": "BPE", "dropout": null, "unk_token": unk_token,
                "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false,
                "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": []
            }
        })
        .to_string()
    }

    fn digest(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    /// Loads a bundle from `tokenizer_json` through the same pinned-file loader the server uses.
    /// The pin takes the vocabulary size the parsed tokenizer reports: the size is not under test.
    fn load_bundle(tokenizer_json: &str) -> TokenizerBundle {
        let dir = tempfile::tempdir().expect("tempdir");
        let files = [
            ("tokenizer.json", tokenizer_json),
            ("tokenizer_config.json", TOKENIZER_CONFIG_JSON),
            ("chat_template.jinja", CHAT_TEMPLATE),
            ("generation_config.json", GENERATION_CONFIG_JSON),
        ];
        for (name, text) in files {
            fs::write(dir.path().join(name), text).expect("write fixture file");
        }
        let vocab_size = Tokenizer::from_bytes(tokenizer_json.as_bytes())
            .expect("parse tokenizer")
            .vocab_size();
        let pin = TokenizerPin {
            tokenizer_json: digest(tokenizer_json.as_bytes()),
            tokenizer_config_json: digest(TOKENIZER_CONFIG_JSON.as_bytes()),
            chat_template_jinja: digest(CHAT_TEMPLATE.as_bytes()),
            generation_config_json: digest(GENERATION_CONFIG_JSON.as_bytes()),
            vocab_size,
        };
        TokenizerBundle::load(dir.path(), &pin).expect("load bundle")
    }

    /// The pinned prompt texts of the five recorded drafts, with their drafts.
    fn pinned_prompts() -> Vec<(String, String)> {
        let doc = cases();
        let rows = doc["cases"].as_array().expect("cases");
        rows.iter()
            .map(|row| {
                let draft = row["draft"].as_str().expect("draft");
                let prompt = row["prompt"].as_str().expect("prompt");
                (draft.to_string(), prompt.to_string())
            })
            .collect()
    }

    /// The ids that a character-level tokenizer gives `<bos>` plus `prompt`.
    fn bos_then_chars(chars: &BTreeMap<char, u32>, prompt: &str) -> Vec<u32> {
        let mut ids = vec![BOS_ID];
        ids.extend(prompt.chars().map(|c| chars[&c]));
        ids
    }

    #[test]
    fn renders_the_pinned_prompt_with_its_recorded_hash() {
        let doc = cases();
        let rows = doc["cases"].as_array().expect("cases");
        assert_eq!(rows.len(), 5, "five pinned drafts");
        for row in rows {
            let draft = row["draft"].as_str().expect("draft");
            let want = row["prompt"].as_str().expect("prompt");
            let got = render(draft);
            assert_eq!(got, want, "prompt text");
            let hash = format!("{:x}", Sha256::digest(got.as_bytes()));
            assert_eq!(
                hash,
                row["prompt_sha256"].as_str().expect("hash"),
                "prompt hash"
            );
        }
    }

    #[test]
    fn the_instruction_and_separator_are_the_pinned_strings() {
        let doc = cases();
        let prompt = doc["cases"][0]["prompt"].as_str().expect("prompt");
        assert!(prompt.starts_with(INSTRUCTION), "instruction prefix");
        assert!(prompt.ends_with(SEPARATOR), "separator suffix");
        assert!(
            !prompt.contains("<bos>"),
            "the rendered text carries no BOS"
        );
        assert!(BOS_TEXT == "<bos>");
    }

    /// `encode` writes the literal BOS text, then the pinned prompt of the recorded draft, and
    /// maps the pair to the BOS id followed by the prompt tokens. The expected ids come from the
    /// recorded prompt text, not from `render`, so the instruction, the trimmed draft (Chinese
    /// text and Unicode padding included) and the separator are each pinned.
    ///
    /// Mutation check: encoding `render(draft)` without the BOS text drops the leading id 2, and
    /// encoding `draft` alone fails every row.
    #[test]
    fn encode_gives_the_bos_id_and_then_the_pinned_prompt_tokens() {
        let prompts = pinned_prompts();
        let texts: Vec<&str> = prompts.iter().map(|(_, prompt)| prompt.as_str()).collect();
        let chars = char_ids(&texts);
        let bundle = load_bundle(&tokenizer_json(&chars, None));
        for (draft, prompt) in &prompts {
            let got = encode(&bundle, draft).expect("encode");
            assert_eq!(got, bos_then_chars(&chars, prompt), "ids of {draft:?}");
        }
    }

    /// The tokenizer's post-processor would add a BOS when special tokens are switched on. The
    /// prompt takes its one BOS from the literal text, so the id 2 occurs exactly once.
    ///
    /// Mutation check: encoding with `add_special_tokens` true in `Tokenizer::encode` gives two
    /// leading BOS ids.
    #[test]
    fn encode_does_not_switch_on_special_tokens_so_bos_appears_once() {
        let chars = char_ids(&[&render("draft")]);
        let bundle = load_bundle(&tokenizer_json(&chars, None));
        let ids = encode(&bundle, "draft").expect("encode");
        assert_eq!(
            ids.iter().filter(|&&id| id == BOS_ID).count(),
            1,
            "ids: {ids:?}"
        );
        assert_eq!(ids.first().copied(), Some(BOS_ID));
    }

    /// A draft with only whitespace around it encodes like the stripped draft, and a blank draft
    /// still gives the BOS id, the instruction and the separator.
    ///
    /// Mutation check: rendering `draft` without `trim()` makes the padded draft longer than
    /// the plain one.
    #[test]
    fn encode_ignores_whitespace_around_the_draft_and_accepts_a_blank_draft() {
        let chars = char_ids(&[&render("Keep it."), &render("")]);
        let bundle = load_bundle(&tokenizer_json(&chars, None));
        let padded = encode(&bundle, " \n\t Keep it.\u{2003}\r\n").expect("padded");
        assert_eq!(padded, encode(&bundle, "Keep it.").expect("plain"));
        let blank = encode(&bundle, " \t\n ").expect("blank");
        assert_eq!(blank, bos_then_chars(&chars, &render("")), "blank draft");
        assert_eq!(blank, encode(&bundle, "").expect("empty"));
    }

    /// A character that the tokenizer cannot represent surfaces as a `Checkpoint` error that
    /// names the encode failure, not as a panic or a prompt with a hole in it.
    ///
    /// Mutation check: swallowing the tokenizer error with `unwrap_or_default()` returns an
    /// empty id list instead of the error.
    #[test]
    fn encode_reports_a_draft_that_the_tokenizer_cannot_encode() {
        let chars = char_ids(&[&render("ascii only")]);
        let bundle = load_bundle(&tokenizer_json(&chars, Some("<unk>")));
        assert!(
            encode(&bundle, "ascii only").is_ok(),
            "every character is in the vocabulary"
        );
        let error = encode(&bundle, "caf\u{e9}").expect_err("a character outside the vocabulary");
        match error {
            EngError::Checkpoint(message) => {
                assert!(
                    message.starts_with("prompt encode failed"),
                    "message: {message}"
                );
            }
            other => panic!("expected Checkpoint, got {other:?}"),
        }
    }

    /// Token ids of the pinned drafts, when the pinned tokenizer is supplied.
    ///
    /// Set GEMMA4_TOKENIZER_DIR to the directory holding tokenizer.json,
    /// tokenizer_config.json, chat_template.jinja and generation_config.json, and the
    /// four SHA-256 hex strings plus the vocabulary size in the matching variables.
    #[test]
    #[ignore = "requires the pinned tokenizer directory and its hashes"]
    fn encodes_the_pinned_token_ids() {
        let Ok(dir) = std::env::var("GEMMA4_TOKENIZER_DIR") else {
            return;
        };
        fn hash(name: &str) -> [u8; 32] {
            let hex = std::env::var(name).unwrap_or_else(|_| panic!("{name} is set"));
            let mut out = [0u8; 32];
            for (index, byte) in out.iter_mut().enumerate() {
                *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).expect("hex");
            }
            out
        }
        let pin = crate::tokenizer::TokenizerPin {
            tokenizer_json: hash("GEMMA4_TOKENIZER_JSON_SHA256"),
            tokenizer_config_json: hash("GEMMA4_TOKENIZER_CONFIG_SHA256"),
            chat_template_jinja: hash("GEMMA4_TEMPLATE_SHA256"),
            generation_config_json: hash("GEMMA4_GENERATION_SHA256"),
            vocab_size: std::env::var("GEMMA4_VOCAB_SIZE")
                .expect("vocab size")
                .parse()
                .expect("vocab number"),
        };
        let bundle =
            TokenizerBundle::load(std::path::Path::new(&dir), &pin).expect("tokenizer bundle");
        for row in cases()["cases"].as_array().expect("cases") {
            let draft = row["draft"].as_str().expect("draft");
            let want: Vec<u32> = row["with_special_tokens"]
                .as_array()
                .expect("ids")
                .iter()
                .map(|v| v.as_u64().expect("id") as u32)
                .collect();
            let got = encode(&bundle, draft).expect("encode");
            assert_eq!(got.first().copied(), Some(2), "BOS id of {}", row["id"]);
            assert_eq!(got, want, "token ids of {}", row["id"]);
        }
    }
}
