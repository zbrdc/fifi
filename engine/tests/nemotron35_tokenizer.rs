use inference_engine::{
    EngError,
    tokenizer::{PromptMessage, Role, Thinking, TokenizerBundle, TokenizerPin, nemotron},
};
use serde_json::{Value, json};
use std::{fs, path::Path};

const BUNDLE_FILES: [&str; 4] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
];

const FIXTURE_TOKENIZER_JSON_SHA256: &str =
    "ecc5ec5e4bc85b4554a526aec0671dded4b8592b73a37a38dac591cadaee2cc1";
const FIXTURE_TOKENIZER_CONFIG_SHA256: &str =
    "28e996652dda4b137a7fa2722c37db8f66a4316eca78b502ef27c6694b4a5602";
const FIXTURE_CHAT_TEMPLATE_SHA256: &str =
    "1362ea23d263faaefdaa56104b3ef95fa75ddabe994daee2866dade55d0db63a";
const FIXTURE_GENERATION_CONFIG_SHA256: &str =
    "6f24a1e60b054fc5055b9eb9236b7e188b0323ae6bece1a5e21912b0b2b0e788";
const FIXTURE_CHAT_TEMPLATE: &str = "{% for message in messages %}<|im_start|>{{ message.content }}<|im_end|>{% endfor %}{% if add_generation_prompt %}<|im_start|>{% endif %}";

fn hex32(text: &str) -> [u8; 32] {
    let mut digest = [0; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).expect("hash hex");
    }
    digest
}

fn fixture_pin() -> TokenizerPin {
    TokenizerPin {
        tokenizer_json: hex32(FIXTURE_TOKENIZER_JSON_SHA256),
        tokenizer_config_json: hex32(FIXTURE_TOKENIZER_CONFIG_SHA256),
        chat_template_jinja: hex32(FIXTURE_CHAT_TEMPLATE_SHA256),
        generation_config_json: hex32(FIXTURE_GENERATION_CONFIG_SHA256),
        vocab_size: 16,
    }
}

fn fixture_tokenizer() -> Vec<u8> {
    let vocab = json!({
        "<unk>": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
        "g": 7,
        "h": 8,
        "i": 9,
        "<|im_start|>": 10,
        "<|im_end|>": 11,
        "j": 12,
        "k": 13,
        "l": 14,
        "m": 15
    });
    serde_json::to_vec(&json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 10, "content": "<|im_start|>", "special": true,
             "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
            {"id": 11, "content": "<|im_end|>", "special": true,
             "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false,
                          "trim_offsets": false, "use_regex": true},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                    "trim_offsets": false, "use_regex": false},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": "<unk>",
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": []
        }
    }))
    .expect("serialize fixture tokenizer")
}

fn write_fixture(dir: &Path) {
    fs::write(dir.join("tokenizer.json"), fixture_tokenizer()).expect("write tokenizer.json");
    fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"tokenizer_class":"TokenizersBackend","eos_token":"<|im_end|>"}"#,
    )
    .expect("write tokenizer_config.json");
    fs::write(dir.join("chat_template.jinja"), FIXTURE_CHAT_TEMPLATE)
        .expect("write chat_template.jinja");
    fs::write(
        dir.join("generation_config.json"),
        r#"{"eos_token_id":[11]}"#,
    )
    .expect("write generation_config.json");
}

fn message(role: Role, content: &str) -> PromptMessage {
    PromptMessage {
        role,
        content: Some(content.to_string()),
        reasoning_content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    }
}

fn decode_rendered(dir: &Path, bundle: &TokenizerBundle, messages: &[PromptMessage]) -> String {
    let ids = bundle
        .encode_prompt(messages, &[], Thinking::Default)
        .expect("render and encode prompt");
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("raw tokenizer loads");
    tokenizer
        .decode(&ids, false)
        .expect("decode rendered prompt")
}

#[test]
fn nemotron35_model_bundle_loads_and_renders_the_trailing_user_turn() {
    let Some(model_dir) = std::env::var_os("NEM35_MODEL_DIR").map(std::path::PathBuf::from) else {
        eprintln!("skip: NEM35_MODEL_DIR is not set");
        return;
    };
    for name in BUNDLE_FILES {
        let path = model_dir.join(name);
        assert!(path.is_file(), "{} must exist", path.display());
        fs::read(&path)
            .unwrap_or_else(|error| panic!("{} must be readable: {error}", path.display()));
    }

    let bundle = nemotron::load(&model_dir).expect("Nemotron bundle loads");
    assert_eq!(bundle.vocab_size(), 131_072);
    assert_eq!(bundle.eos_token_id(), 11);

    let config: Value = serde_json::from_slice(
        &fs::read(model_dir.join("tokenizer_config.json")).expect("config readable"),
    )
    .expect("tokenizer config parses");
    let eos = config["eos_token"].as_str().expect("eos_token string");
    assert_eq!(bundle.tokenizer().token_id(eos), Some(11));
    let raw_tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .expect("raw tokenizer loads");
    assert_eq!(raw_tokenizer.decode(&[11], false).expect("decode EOS"), eos);

    let text = "Nemotron runs on the Rust engine.";
    let encoded = bundle
        .tokenizer()
        .encode(text)
        .expect("encode round-trip text");
    assert_eq!(
        bundle
            .tokenizer()
            .decode(&encoded)
            .expect("decode round-trip text"),
        text
    );

    let messages = [
        message(Role::User, "First user turn."),
        message(Role::Assistant, "First assistant reply."),
        message(Role::User, "Second user turn."),
        message(Role::Assistant, "Second assistant reply."),
        message(Role::User, "Trailing user turn."),
    ];
    let rendered = raw_tokenizer
        .decode(
            &bundle
                .encode_prompt(&messages, &[], Thinking::Default)
                .expect("conversation renders and encodes"),
            false,
        )
        .expect("decode rendered conversation");
    for marker in ["<|im_start|>", "<|im_end|>", "<think>"] {
        assert!(rendered.contains(marker), "rendered prompt lacks {marker}");
    }
    for turn in [
        "First user turn.",
        "First assistant reply.",
        "Second user turn.",
        "Second assistant reply.",
        "Trailing user turn.",
    ] {
        assert!(rendered.contains(turn), "rendered prompt lacks {turn:?}");
    }
}

#[test]
fn nemotron35_fixture_bundle_uses_the_same_constructor_path() {
    let dir = tempfile::tempdir().expect("fixture directory");
    write_fixture(dir.path());
    let bundle = TokenizerBundle::load(dir.path(), &fixture_pin()).expect("fixture bundle loads");
    assert_eq!(bundle.vocab_size(), 16);
    assert_eq!(bundle.eos_token_id(), 11);
    assert_eq!(
        bundle
            .tokenizer()
            .decode(&bundle.tokenizer().encode("abc").unwrap())
            .unwrap(),
        "abc"
    );

    let messages = [
        message(Role::User, "a"),
        message(Role::Assistant, "b"),
        message(Role::User, "c"),
    ];
    let rendered = decode_rendered(dir.path(), &bundle, &messages);
    assert_eq!(
        rendered,
        "<|im_start|>a<|im_end|><|im_start|>b<|im_end|><|im_start|>c<|im_end|><|im_start|>"
    );
}

#[test]
fn nemotron35_missing_bundle_member_returns_checkpoint_error() {
    let dir = tempfile::tempdir().expect("missing bundle directory");
    let error = match nemotron::load(dir.path()) {
        Err(error) => error,
        Ok(_) => panic!("empty directory unexpectedly loaded"),
    };
    match error {
        EngError::Checkpoint(message) => {
            assert!(message.contains("tokenizer.json"), "{message}");
            assert!(message.contains("missing"), "{message}");
        }
        other => panic!("expected Checkpoint, got {other:?}"),
    }

    for missing in BUNDLE_FILES {
        let dir = tempfile::tempdir().expect("fixture directory");
        write_fixture(dir.path());
        fs::remove_file(dir.path().join(missing)).expect("remove bundle member");

        let error = match TokenizerBundle::load(dir.path(), &fixture_pin()) {
            Err(error) => error,
            Ok(_) => panic!("missing {missing} unexpectedly loaded"),
        };
        match error {
            EngError::Checkpoint(message) => {
                assert!(message.contains(missing), "{message}");
                assert!(message.contains("missing"), "{message}");
            }
            other => panic!("expected Checkpoint, got {other:?}"),
        }
    }
}
