//! Mac reference rows for the tokenizer seams.
//!
//! These tests skip unless the pinned checkpoint directory is present.
//! They cover S1-T6 file pins, S2-T8 golden id vectors, S2-T9 eos ids,
//! and S4-T5 the rendered smoke prompt vector. They load no model and use no GPU.
use inference_engine::tokenizer::{
    DecodeBatch, DecodeFinish, IncrementalDecoder, OutputAssembler, PromptMessage, Role, Thinking,
    TokenizerBundle, TokenizerPin, ToolCallSpec, ToolSpec,
};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const GOLDENS_JSON: &str = include_str!("../testdata/tokenizer-goldens.json");
const TOKENIZER_JSON_SHA256: &str =
    "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4";
const TOKENIZER_CONFIG_SHA256: &str =
    "e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182";
const CHAT_TEMPLATE_SHA256: &str =
    "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259";
// Hash from `engine/testdata/mac-phase0a.json` (generation_config.json, 202 bytes).
const GENERATION_CONFIG_SHA256: &str =
    "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e";

fn goldens() -> serde_json::Value {
    serde_json::from_str(GOLDENS_JSON).expect("goldens JSON parses")
}

fn hex32(text: &str) -> [u8; 32] {
    let bytes = text.as_bytes();
    let mut out = [0u8; 32];
    for (index, slot) in out.iter_mut().enumerate() {
        let pair = &bytes[index * 2..index * 2 + 2];
        *slot = u8::from_str_radix(std::str::from_utf8(pair).expect("hex"), 16).expect("hex digit");
    }
    out
}

fn file_digest(path: &PathBuf) -> [u8; 32] {
    let bytes = std::fs::read(path).expect("file reads");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

fn tokenizer_dir() -> Option<PathBuf> {
    match std::env::var("INFERENCE_TOKENIZER_DIR") {
        Ok(value) => Some(PathBuf::from(value)),
        Err(_) => {
            let default = PathBuf::from("/Users/dan/edge0/models/edge0-35b");
            default.exists().then_some(default)
        }
    }
}

fn pinned_bundle() -> Option<(PathBuf, TokenizerBundle)> {
    let dir = tokenizer_dir()?;
    let goldens = goldens();
    let reference = &goldens["reference"];
    let pin = TokenizerPin {
        tokenizer_json: hex32(TOKENIZER_JSON_SHA256),
        tokenizer_config_json: hex32(TOKENIZER_CONFIG_SHA256),
        chat_template_jinja: hex32(CHAT_TEMPLATE_SHA256),
        generation_config_json: hex32(GENERATION_CONFIG_SHA256),
        vocab_size: reference["vocab_size"].as_u64().expect("vocab_size") as u32,
    };
    let bundle = TokenizerBundle::load(&dir, &pin).expect("bundle loads");
    Some((dir, bundle))
}

fn skip_without_pin() {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        panic!(
            "the Mac tokenizer qualification requires INFERENCE_TOKENIZER_DIR or the pinned model path"
        );
    }
    eprintln!("skip: INFERENCE_TOKENIZER_DIR is unavailable");
}

fn user(content: &str) -> PromptMessage {
    PromptMessage {
        role: Role::User,
        content: Some(content.to_string()),
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    }
}

#[test]
fn prefix_stability_repeats_on_the_real_tokenizer() {
    let Some((_dir, bundle)) = pinned_bundle() else {
        skip_without_pin();
        return;
    };
    let contents = ["one", "two", "three", "four"];
    let mut streams: Vec<Vec<u32>> = Vec::new();
    for count in 1..=contents.len() {
        let messages: Vec<PromptMessage> = contents[..count].iter().map(|c| user(c)).collect();
        streams.push(
            bundle
                .encode_prompt(&messages, &[], Thinking::Default)
                .expect("encode prompt"),
        );
    }
    let tail = shared_suffix_len(&streams);
    for pair in streams.windows(2) {
        let short = &pair[0][..pair[0].len() - tail];
        let long = &pair[1][..pair[1].len() - tail];
        assert!(
            long.starts_with(short),
            "the message body must extend the prompt tokens"
        );
    }
}

fn shared_suffix_len(streams: &[Vec<u32>]) -> usize {
    let mut length = 0;
    while streams.iter().all(|stream| {
        stream.len() > length + 1
            && stream[stream.len() - 1 - length] == streams[0][streams[0].len() - 1 - length]
    }) {
        length += 1;
    }
    length
}

#[test]
fn checkpoint_files_match_the_pinned_hashes() {
    let Some((dir, _bundle)) = pinned_bundle() else {
        skip_without_pin();
        return;
    };
    let expected = [
        ("tokenizer.json", TOKENIZER_JSON_SHA256),
        ("tokenizer_config.json", TOKENIZER_CONFIG_SHA256),
        ("chat_template.jinja", CHAT_TEMPLATE_SHA256),
        ("generation_config.json", GENERATION_CONFIG_SHA256),
    ];
    for (name, digest) in expected {
        let hex: String = file_digest(&dir.join(name))
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(hex, digest, "{name} hash");
    }
}

#[test]
fn golden_id_vectors_match_the_real_tokenizer() {
    let Some((_dir, bundle)) = pinned_bundle() else {
        skip_without_pin();
        return;
    };
    let goldens = goldens();
    for row in goldens["encodes"].as_array().expect("encodes array") {
        let id = row["id"].as_str().expect("id");
        let text = row["text"].as_str().expect("text");
        let expected: Vec<u32> = row["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|value| value.as_u64().expect("id value") as u32)
            .collect();
        let ids = bundle.tokenizer().encode(text).expect("encode");
        assert_eq!(ids, expected, "golden id vector {id}");
    }
}

#[test]
fn eos_token_id_matches_the_reference() {
    let Some((_dir, bundle)) = pinned_bundle() else {
        skip_without_pin();
        return;
    };
    let goldens = goldens();
    let expected = goldens["reference"]["eos_token_id"]
        .as_u64()
        .expect("eos_token_id") as u32;
    assert_eq!(bundle.eos_token_id(), expected);
}

// The pinned generation_config.json lists eos_token_id [248046, 248044].
// mlx-lm 0.31.0 builds its stop set from that list (else config.json, else only
// the tokenizer eos_token `<|im_end|>`, id 248046). The engine returns the union
// of the tokenizer eos_token and the list, which is the same set here.
#[test]
fn eos_token_ids_match_the_reference_loader() {
    let Some((_dir, bundle)) = pinned_bundle() else {
        skip_without_pin();
        return;
    };
    assert_eq!(bundle.eos_token_ids(), [248_046, 248_044]);
}

#[test]
fn smoke_prompt_matches_the_reference_vector() {
    let Some((_dir, bundle)) = pinned_bundle() else {
        skip_without_pin();
        return;
    };
    let goldens = goldens();
    let smoke = &goldens["smoke"];
    let text = smoke["text"].as_str().expect("smoke text");
    let expected: Vec<u32> = smoke["prompt_ids"]
        .as_array()
        .expect("prompt_ids")
        .iter()
        .map(|value| value.as_u64().expect("id value") as u32)
        .collect();
    let ids = bundle
        .encode_prompt(&[user(text)], &[], Thinking::Default)
        .expect("encode prompt");
    assert_eq!(ids, expected, "smoke prompt vector");
}

fn golden_ids(row: &serde_json::Value) -> Vec<u32> {
    row["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|value| u32::try_from(value.as_u64().expect("id value")).expect("id fits u32"))
        .collect()
}

// Streams each golden id vector and EOS through the assembler. The joined deltas
// must equal the one-shot decode of the same ids, including the CJK row whose
// characters split across byte-level tokens.
#[test]
fn streamed_output_equals_the_one_shot_decode() {
    let Some((_dir, bundle)) = pinned_bundle() else {
        skip_without_pin();
        return;
    };
    let goldens = goldens();
    for row in goldens["encodes"].as_array().expect("encodes array") {
        let id = row["id"].as_str().expect("id");
        let ids = golden_ids(row);
        let expected = bundle.tokenizer().decode(&ids).expect("one-shot decode");
        // This loop mirrors `Executor::decode`: one batch per token, then the
        // decoder tail on the EOS batch. S8-T4 replaces it with the real executor.
        let mut decoder = IncrementalDecoder::new(bundle.tokenizer());
        let mut assembler =
            OutputAssembler::new(vec!["<<never-generated>>".to_string()]).expect("assembler");
        let mut streamed = String::new();
        for token in ids {
            let text = decoder.push(token).expect("decode");
            let batch = DecodeBatch {
                tokens: vec![token],
                text,
                finish: None,
            };
            let delta = assembler.push_batch(&batch).expect("push");
            streamed.push_str(&delta.text);
        }
        let batch = DecodeBatch {
            tokens: vec![bundle.eos_token_id()],
            text: decoder.finish().expect("tail"),
            finish: Some(DecodeFinish::Stop),
        };
        let last = assembler.push_batch(&batch).expect("eos");
        streamed.push_str(&last.text);
        assert_eq!(last.finish, Some(DecodeFinish::Stop), "{id}");
        assert_eq!(streamed, expected, "streamed text for {id}");
    }
}

// The shared tokenizer keeps decoder creation and each push cheap. The old
// decoder cloned the 20 MB tokenizer per job and rebuilt the vocabulary map on
// every token. The bounds are far above the measured cost, far below the old one.
#[test]
fn decoder_creation_and_push_stay_cheap_on_the_real_tokenizer() {
    let Some((_dir, bundle)) = pinned_bundle() else {
        skip_without_pin();
        return;
    };
    let started = std::time::Instant::now();
    let decoders: Vec<IncrementalDecoder> = (0..100)
        .map(|_| IncrementalDecoder::new(bundle.tokenizer()))
        .collect();
    let creation = started.elapsed();
    eprintln!("100 decoder creations: {creation:?}");
    assert!(
        creation < std::time::Duration::from_millis(100),
        "{creation:?}"
    );
    let goldens = goldens();
    let ids = golden_ids(&goldens["encodes"][1]);
    let mut decoder = decoders.into_iter().next().expect("decoder");
    let started = std::time::Instant::now();
    for &token in &ids {
        decoder.push(token).expect("push");
    }
    let pushes = started.elapsed();
    eprintln!("{} pushes: {pushes:?}", ids.len());
    let per_push = pushes / u32::try_from(ids.len()).expect("count");
    assert!(
        per_push < std::time::Duration::from_millis(2),
        "{per_push:?}"
    );
}

const TOJSON_GOLDENS_JSON: &str = include_str!("../src/tokenizer/testdata/tojson-goldens.json");

fn golden_message(row: &serde_json::Value) -> PromptMessage {
    let role = match row["role"].as_str().expect("role") {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        _ => Role::Tool,
    };
    let calls = row["tool_calls"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    PromptMessage {
        role,
        content: row["content"].as_str().map(str::to_string),
        name: None,
        tool_call_id: row["tool_call_id"].as_str().map(str::to_string),
        tool_calls: calls
            .iter()
            .map(|call| ToolCallSpec {
                id: call["id"].as_str().expect("id").to_string(),
                name: call["name"].as_str().expect("name").to_string(),
                arguments: call["arguments"].to_string(),
            })
            .collect(),
    }
}

// Decision 0029: the checkpoint template, rendered with OpenAI tools as sent,
// must match transformers 5.17.0 byte for byte, and the prompt ids must match
// the ids of that reference text.
#[test]
fn tool_prompts_match_the_reference_on_the_real_bundle() {
    let Some((_dir, bundle)) = pinned_bundle() else {
        skip_without_pin();
        return;
    };
    let goldens: serde_json::Value = serde_json::from_str(TOJSON_GOLDENS_JSON).expect("goldens");
    for row in goldens["renders"].as_array().expect("renders") {
        let id = row["id"].as_str().expect("id");
        let messages: Vec<serde_json::Value> =
            serde_json::from_str(row["messages_json"].as_str().expect("messages")).expect("json");
        let tools: Vec<serde_json::Value> =
            serde_json::from_str(row["tools_json"].as_str().expect("tools")).expect("json");
        let thinking = match row["thinking"].as_str().expect("thinking") {
            "on" => Thinking::On,
            "off" => Thinking::Off,
            _ => Thinking::Default,
        };
        let prompts: Vec<PromptMessage> = messages.iter().map(golden_message).collect();
        let specs: Vec<ToolSpec> = tools.into_iter().map(ToolSpec).collect();
        let ids = bundle
            .encode_prompt(&prompts, &specs, thinking)
            .expect("encode prompt");
        let reference = row["reference"].as_str().expect("reference");
        let expected = bundle
            .tokenizer()
            .encode(reference)
            .expect("encode reference");
        assert_eq!(ids, expected, "prompt ids for {id}");
    }
}
