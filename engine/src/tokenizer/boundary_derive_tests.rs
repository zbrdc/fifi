//! Boundary-derivation equivalence tests. The derivation must be EXACT or
//! fall back to the three-pass path — never approximate in the result.

use super::{PromptMessage, Role, Thinking, TokenizerBundle, TokenizerPin};
use sha2::{Digest, Sha256};
use std::path::Path;

fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// The tiny in-repo fixture (same strings as tests.rs) for fast units.
/// The real Qwen3.6 model dir for on-disk corpus tests: the
/// `FIFI_QWEN36_MODEL_DIR` env override (a directory, or a
/// `tokenizer.json` file — its parent is used) when set, else the
/// default path. Returns None (and prints why) when the dir is absent, so
/// hosts without the weights SKIP these tests instead of failing them.
pub(crate) fn qwen_model_dir() -> Option<std::path::PathBuf> {
    let dir = match std::env::var_os("FIFI_QWEN36_MODEL_DIR") {
        Some(v) => {
            let p = std::path::PathBuf::from(v);
            if p.is_file() {
                p.parent().map(std::path::Path::to_path_buf).unwrap_or(p)
            } else {
                p
            }
        }
        None => std::path::PathBuf::from("/media/dan/nvme/models/qwen3.6-35b-a3b"),
    };
    let present = dir.join("tokenizer.json").is_file();
    if !present && std::env::var_os("FIFI_QWEN36_MODEL_DIR").is_some() {
        panic!(
            "FIFI_QWEN36_MODEL_DIR is set but {:?} has no tokenizer.json — \
             a typo silently skipped every real-model test",
            dir
        );
    }
    if !present {
        eprintln!(
            "qwen model dir {:?} not on disk: SKIPPING real-model tests",
            dir
        );
    }
    present.then_some(dir)
}

pub(crate) const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":100,"content":" world","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":101,"content":"<|end|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"hello":8,"a":9,"b":10,"he":11,"hel":12,"hell":13},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;
const CONFIG_JSON: &str = r#"{"tokenizer_class":"TestTokenizer","eos_token":"<|end|>"}"#;
const GEN_JSON: &str = r#"{"bos_token_id":15,"eos_token_id":[15,11],"temperature":1.0}"#;
/// A template whose generation prompt starts with the atomic special token
/// `<|end|>`: the no-merge boundary condition the derivation relies on.
pub(crate) const TEMPLATE_SPECIAL: &str = concat!(
    r#"{% for m in messages %}{{ m.role }}:{{ m.content }}{% endfor %}"#,
    r#"{% if add_generation_prompt %}<|end|>hello{% endif %}"#,
);
/// A template whose generation prompt starts with plain BPE text (no added
/// token at the boundary): the derivation MUST fall back — accepting it
/// would trust a boundary a merge can cross.
pub(crate) const TEMPLATE_PLAIN: &str = concat!(
    r#"{% for m in messages %}{{ m.role }}:{{ m.content }}{% endfor %}"#,
    r#"{% if add_generation_prompt %}hello{% endif %}"#,
);
/// A template with text AFTER the generation prompt: the suffix is followed
/// by more template text, so the prompt ids do not end with the suffix ids
/// and the ends_with guard must route to the fallback.
const TEMPLATE_WRAPPED: &str = concat!(
    r#"{% for m in messages %}{{ m.role }}:{{ m.content }}{% endfor %}"#,
    r#"{% if add_generation_prompt %}<|end|>hello{% endif %}ab"#,
);

/// A template whose generation block is CONTENT-CONDITIONAL. The probe
/// suffix observed for [user "x"] is `<|end|>hello`, and user content
/// `hello<|end|>hello` makes every alignment guard pass while the marker
/// sits inside USER CONTENT, not at a generation boundary. The old method
/// returns no tip there (head equals full); the derivation must NEVER claim
/// exactness in that request — derive exactly or fall back.
const TEMPLATE_CONDITIONAL: &str = r#"{{ messages[0].content }}{% if add_generation_prompt and messages[0].content == 'x' %}<|end|>hello{% endif %}"#;
/// The same vocabulary with the boundary token lstrip=true — the matcher
/// may consume preceding whitespace, so the added-token split is not
/// byte-faithful at the seam. Membership alone must not count as
/// isolation.
pub(crate) const TOKENIZER_JSON_LSTRIP: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":100,"content":" world","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":101,"content":"<|end|>","special":true,"single_word":false,"lstrip":true,"rstrip":false,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"hello":8,"a":9,"b":10,"he":11,"hel":12,"hell":13},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;

const TOKENIZER_JSON_WORD_START: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":100,"content":" world","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":101,"content":"<|end|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":true},{"id":102,"content":"hellox","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":103,"content":"mark","special":false,"single_word":true,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"hello":8,"a":9,"b":10,"he":11,"hel":12,"hell":13},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;

const TOKENIZER_JSON_STRIP_OX: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":100,"content":" world","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":101,"content":"<|end|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":{"type":"Replace","pattern":{"String":"ox"},"content":""},"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"hello":8,"a":9,"b":10,"he":11,"hel":12,"hell":13},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;

const TOKENIZER_JSON_RSTRIP: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":100,"content":" world","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":101,"content":"<|end|>","special":true,"single_word":false,"lstrip":false,"rstrip":true,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"hello":8,"a":9,"b":10,"he":11,"hel":12,"hell":13},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;

const TOKENIZER_JSON_PADDED: &str = r#"{"version":"1.0","truncation":null,"padding":{"strategy":{"Fixed":10},"direction":"Right","pad_to_multiple_of":null,"pad_token":"hello","pad_id":8,"pad_type_id":0},"added_tokens":[{"id":100,"content":" world","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":101,"content":"<|end|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"hello":8,"a":9,"b":10,"he":11,"hel":12,"hell":13},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;

const TOKENIZER_JSON_NORMALIZED: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":100,"content":" world","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":101,"content":"<|end|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":true}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"hello":8,"a":9,"b":10,"he":11,"hel":12,"hell":13},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;
const TOKENIZER_JSON_SINGLE_WORD: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":100,"content":" world","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":101,"content":"<|end|>","special":true,"single_word":true,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"hello":8,"a":9,"b":10,"he":11,"hel":12,"hell":13},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;
const TOKENIZER_JSON_HEAD_SINGLE_WORD: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":100,"content":" world","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":101,"content":"<|end|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":15,"content":"aa","special":false,"single_word":true,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"hello":8,"a":9,"b":10,"he":11,"hel":12,"hell":13},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;
const TOKENIZER_JSON_TRUNCATED: &str = r#"{"version":"1.0","truncation":{"direction":"Left","max_length":4,"strategy":"LongestFirst","stride":0},"padding":null,"added_tokens":[{"id":100,"content":" world","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":101,"content":"<|end|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"hello":8,"a":9,"b":10,"he":11,"hel":12,"hell":13},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;
const TOKENIZER_JSON_TRUNCATED_WIDE: &str = r#"{"version":"1.0","truncation":{"direction":"Left","max_length":64,"strategy":"LongestFirst","stride":0},"padding":null,"added_tokens":[{"id":100,"content":" world","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":101,"content":"<|end|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"hello":8,"a":9,"b":10,"he":11,"hel":12,"hell":13},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;

fn lstrip_tokenizer_json() -> String {
    TOKENIZER_JSON_LSTRIP.to_string()
}

/// lstrip repro template: content only, no role loop, so `hello ` keeps
/// its trailing space in the head render (`[8, 4]`) while the lstrip
/// matcher eats it in the full render (`[8, 15, 8]`).
const TEMPLATE_CONTENT: &str =
    r#"{{ messages[0].content }}{% if add_generation_prompt %}<|end|>hello{% endif %}"#;

pub(crate) fn fixture_bundle(template: &str) -> (tempfile::TempDir, TokenizerBundle) {
    fixture_bundle_with(TOKENIZER_JSON, template)
}

pub(crate) fn fixture_bundle_with(
    tokenizer_json: &str,
    template: &str,
) -> (tempfile::TempDir, TokenizerBundle) {
    let dir = tempfile::tempdir().expect("tempdir");
    let write = |n: &str, b: &str| fs_write(dir.path(), n, b);
    write("tokenizer.json", tokenizer_json);
    write("tokenizer_config.json", CONFIG_JSON);
    write("chat_template.jinja", template);
    write("generation_config.json", GEN_JSON);
    let pin = TokenizerPin {
        tokenizer_json: digest(tokenizer_json.as_bytes()),
        tokenizer_config_json: digest(CONFIG_JSON.as_bytes()),
        chat_template_jinja: digest(template.as_bytes()),
        generation_config_json: digest(GEN_JSON.as_bytes()),
        // The pin reflects the ACTUAL fixture vocabulary, so variant
        // fixtures that add tokens keep loading.
        vocab_size: tokenizers::Tokenizer::from_bytes(tokenizer_json.as_bytes())
            .expect("fixture tokenizer")
            .get_vocab_size(true)
            .try_into()
            .expect("vocab size fits u32"),
    };
    let bundle = TokenizerBundle::load(dir.path(), &pin).expect("bundle");
    (dir, bundle)
}

fn fs_write(dir: &Path, name: &str, body: &str) {
    std::fs::write(dir.join(name), body).expect("write fixture");
}

pub(crate) fn msgs(pairs: &[(&str, &str)]) -> Vec<PromptMessage> {
    pairs
        .iter()
        .map(|(role, content)| PromptMessage {
            role: match *role {
                "system" => Role::System,
                "tool" => Role::Tool,
                "user" => Role::User,
                _ => Role::Assistant,
            },
            content: Some(content.to_string()),
            reasoning_content: None,
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        })
        .collect()
}

/// The derived boundaries equal the three-pass boundaries on the fixture
/// template whose generation prompt starts with an atomic token.
#[test]
fn derived_tip_equals_three_pass_on_atomic_boundary() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_SPECIAL);
    let messages = msgs(&[("user", "hello world"), ("assistant", "hello")]);
    for thinking in [Thinking::Default, Thinking::On, Thinking::Off] {
        let three = bundle
            .encode_prompt_with_boundary(&messages, &[], thinking)
            .expect("three-pass");
        let (ids, derived, exact) = bundle
            .encode_prompt_with_derived_boundary(&messages, &[], thinking)
            .expect("derived");
        assert!(exact, "atomic boundary must derive exactly");
        assert_eq!(three.0, ids);
        assert_eq!(three.1.tip, derived.tip, "tip for {thinking:?}");
        assert_eq!(three.1.root, derived.root, "root for {thinking:?}");
    }
}

/// A corrupted suffix id forces the fallback, and the fallback result is
/// the three-pass result, byte for byte.
#[test]
fn corrupted_suffix_falls_back_to_three_pass() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_SPECIAL);
    let messages = msgs(&[("user", "hello world"), ("assistant", "hello")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    // A clean-starting forced suffix that does not match the tail. NOTE:
    // tokenizers remaps fixture added-token ids on load — <|end|> becomes
    // id 15, not the JSON's 101 — so query the real id, never hardcode it.
    let end_id = bundle
        .generation_suffix_ids_for(&messages, &[], Thinking::Default)
        .and_then(|s| s.first().copied())
        .expect("real suffix starts with the boundary token");
    let bad_suffix = [end_id, end_id, end_id];
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary_forced(&messages, &[], Thinking::Default, &bad_suffix)
        .expect("derived");
    assert!(!exact, "a non-matching suffix must not claim exactness");
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip);
    assert_eq!(three.1.root, derived.root);
}

/// Off-by-one sensitivity: a suffix one token short of the truth must NOT
/// be accepted (the ends_with check fails) — and falls back — so an
/// off-by-one derivation cannot silently ship.
#[test]
fn off_by_one_suffix_is_rejected() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_SPECIAL);
    let messages = msgs(&[("user", "hello"), ("assistant", "hello")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    let true_suffix = bundle
        .generation_suffix_ids_for(&messages, &[], Thinking::Default)
        .expect("real suffix");
    let short = &true_suffix[..true_suffix.len() - 1];
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary_forced(&messages, &[], Thinking::Default, short)
        .expect("derived");
    // The one-token-short suffix does NOT end-with match (the prompt's last
    // token is the suffix's dropped token), so it must fall back and return
    // the three-pass result — an off-by-one derivation cannot ship.
    assert!(!exact, "a one-token-short suffix must not derive");
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip);
    assert_eq!(three.1.root, derived.root);
}

/// A non-atomic boundary (plain BPE text at the generation prompt start)
/// must NOT derive: the atomicity guard routes it to the three-pass
/// fallback, and the fallback result equals the three-pass result.
#[test]
fn non_atomic_boundary_falls_back_to_three_pass() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_PLAIN);
    let messages = msgs(&[("user", "hello world"), ("assistant", "hello")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    assert!(!exact, "a plain-text boundary must not claim exactness");
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip);
    assert_eq!(three.1.root, derived.root);
}

/// A stray clean suffix on a template whose generation block is followed by
/// more template text: the wrapped template's own suffix legitimately
/// includes that trailing text (per-request derivation handles it), so this
/// test forces a suffix the prompt does NOT end with — atomicity alone must
/// not be enough, the ends_with guard has to refuse and fall back.
#[test]
fn wrapped_generation_prompt_falls_back_to_three_pass() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_WRAPPED);
    let messages = msgs(&[("user", "hello world"), ("assistant", "hello")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    // An ATOMIC suffix that the prompt does not end with: atomicity alone
    // must not be enough — the ends_with guard has to refuse and fall back.
    // <|end|> loads as id 15 (tokenizers remaps fixture ids); the wrapped
    // template has no derivable suffix (head is not a byte-prefix), so take
    // the boundary id from a direct encode.
    let end_id = bundle
        .tokenizer()
        .encode("<|end|>")
        .expect("boundary token encodes")[0];
    let stray_atomic_suffix = [end_id, end_id, end_id];
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary_forced(
            &messages,
            &[],
            Thinking::Default,
            &stray_atomic_suffix,
        )
        .expect("derived");
    assert!(
        !exact,
        "a suffix the prompt does not end with must not claim exactness"
    );
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip);
    assert_eq!(three.1.root, derived.root);
}

/// The real-corpus equivalence: recorded Claude Code conversation shapes
/// (pure-append prefixes, tool results, tool_use, thinking) rendered with
/// the REAL pinned template and tokenizer; requires the model dir (fails
/// loudly when absent — the test host provides it).
#[test]
fn derived_boundaries_match_three_pass_on_the_real_corpus() {
    // The model dir resolves through the shared helper (env override or
    // the default path); absent means SKIP with a printed reason so
    // portable hosts (the Mac) run everything else meaningfully.
    let Some(dir) = qwen_model_dir() else {
        return;
    };
    let read = |n: &str| std::fs::read(dir.join(n)).unwrap();
    let pin = TokenizerPin {
        tokenizer_json: digest(&read("tokenizer.json")),
        tokenizer_config_json: digest(&read("tokenizer_config.json")),
        chat_template_jinja: digest(&read("chat_template.jinja")),
        generation_config_json: digest(&read("generation_config.json")),
        vocab_size: 248077,
    };
    let bundle = TokenizerBundle::load(&dir, &pin).expect("real bundle");

    // The corpus carries ANTHROPIC WIRE requests (system prompt, real
    // tool_use ids/names/input, tool_result references, thinking blocks,
    // input_schema tools). Every conversation goes through the PRODUCTION
    // converter — no hand adapter anywhere.
    let corpus: serde_json::Value =
        serde_json::from_str(include_str!("testdata/boundary-corpus.json")).expect("corpus");
    // At least one conversation carries a merged assistant message with a
    // thinking block AND a tool_use block (the production CC shape), and
    // the thinking payload must be a NON-EMPTY string — an absent, empty,
    // or non-string payload parses as absent thinking, which would make
    // the merged-case pin below vacuous.
    assert!(
        corpus["conversations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|conv| {
                conv["messages"].as_array().unwrap().iter().any(|m| {
                    m["role"] == "assistant"
                        && m["content"].as_array().is_some_and(|blocks| {
                            blocks.iter().any(|b| {
                                b["type"] == "thinking"
                                    && b["thinking"].as_str().is_some_and(|t| !t.is_empty())
                            }) && blocks.iter().any(|b| b["type"] == "tool_use")
                        })
                })
            }),
        "corpus must exercise a merged thinking+tool_use assistant message"
    );
    let mut checked = 0;
    let mut converted_merged = false;
    for conv in corpus["conversations"].as_array().unwrap() {
        let body = serde_json::json!({
            "model": "qwen3.6-35b-a3b",
            "max_tokens": 16,
            "system": conv["system"],
            "messages": conv["messages"],
            "tools": conv["tools"],
        });
        let request =
            crate::server::anthropic::parse_anthropic(&serde_json::to_vec(&body).unwrap())
                .expect("production anthropic parse");
        let messages = request.input.messages;
        let tools = request.input.tools;
        assert!(
            messages
                .first()
                .is_some_and(|m| matches!(m.role, Role::System)),
            "corpus case {} must open with a system message (production parse)",
            conv["name"]
        );
        // The merged thinking+tool_use assistant message must survive the
        // production parse with BOTH payloads: non-empty reasoning_content
        // and its tool calls. Only real-full carries the merged wire
        // shape, so the pin is corpus-wide.
        converted_merged |= messages.iter().any(|m| {
            matches!(m.role, Role::Assistant)
                && m.reasoning_content
                    .as_deref()
                    .is_some_and(|t| !t.is_empty())
                && !m.tool_calls.is_empty()
        });
        for thinking in [Thinking::Default, Thinking::On, Thinking::Off] {
            let three = bundle
                .encode_prompt_with_boundary(&messages, &tools, thinking)
                .expect("three-pass");
            let (ids, derived, exact) = bundle
                .encode_prompt_with_derived_boundary(&messages, &tools, thinking)
                .expect("derived");
            assert_eq!(three.0, ids, "ids must never differ");
            assert_eq!(
                three.1.tip, derived.tip,
                "tip {} {thinking:?}",
                conv["name"]
            );
            assert_eq!(
                three.1.root, derived.root,
                "root {} {thinking:?}",
                conv["name"]
            );
            // Every real Qwen corpus case must DERIVE — this is the
            // proof the fast path fires on the real template, not just the
            // equivalence of both paths.
            assert!(
                exact,
                "corpus case {} must derive on the real template ({thinking:?})",
                conv["name"]
            );
            // A system opener is the conversation root: the root boundary
            // must exist for these real shapes.
            assert!(
                derived.root.is_some_and(|r| r > 0),
                "corpus case {} must have a nonempty root ({thinking:?})",
                conv["name"]
            );
            checked += 1;
        }
    }
    assert!(
        converted_merged,
        "the merged thinking+tool_use assistant message must parse with reasoning_content and tool_calls"
    );
    // 5 conversations x 3 thinking modes, exactly — a dropped or added
    // corpus case must fail loudly, not silently shrink coverage.
    assert_eq!(checked, 15, "corpus must exercise every conversation");
}

/// Regression: the conditional template with user content carrying the end
/// token. Every alignment guard passes at the CURRENT tree (the probe
/// suffix <|end|>hello matches the tail), the marker sits inside user
/// content, and the old method yields no tip (head equals full render). The
/// derivation must report derived=false and return the old result.
#[test]
fn end_token_inside_user_content_never_derives() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_CONDITIONAL);
    let messages = msgs(&[("user", "hello<|end|>hello")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    assert_eq!(three.1.tip, None, "old method: no tip (head == full)");
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    assert!(!exact, "a marker inside user content must never derive");
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip, "tip must equal the old result");
    assert_eq!(three.1.root, derived.root);
}

/// Regression: lstrip=true on the boundary token lets the matcher eat
/// head whitespace (user content `hello ` loses its trailing space), so
/// the added-token split is not byte-faithful. Must fall back.
#[test]
fn lstrip_added_token_never_derives() {
    let (_d, bundle) = fixture_bundle_with(&lstrip_tokenizer_json(), TEMPLATE_CONTENT);
    let messages = msgs(&[("user", "hello ")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    assert_eq!(three.1.tip, None, "old method: head ids are not a prefix");
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    assert!(!exact, "a whitespace-consuming token must never derive");
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip);
    assert_eq!(three.1.root, derived.root);
}

/// Coverage: system/tool roles and multi-message conversations through
/// the derived path WITHOUT suffix_override, counting that the suite sees
/// both derived and fallback outcomes on the fixture template.
#[test]
fn system_and_tool_roles_cover_the_derived_path() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_SPECIAL);
    let mut messages = msgs(&[("system", "sys prompt"), ("user", "hello world")]);
    messages.push(PromptMessage {
        role: Role::Tool,
        content: Some("tool output".to_string()),
        reasoning_content: None,
        name: None,
        tool_call_id: Some("call_1".to_string()),
        tool_calls: Vec::new(),
    });
    messages.extend(msgs(&[("user", "hello")]));
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip);
    assert_eq!(three.1.root, derived.root);
    // A system message is the conversation root: the root boundary must
    // be nonempty for these roles, pinning that the boundary exists at all.
    assert!(derived.root.is_some_and(|r| r > 0), "root must be nonempty");
    // The special template's generation block is a trailing atomic block
    // for these roles too — the fast path must fire.
    assert!(exact, "system/tool/multi-message conversations must derive");
}

/// Seam guard: a forced suffix the ids genuinely end with, at a VALID
/// candidate index, but whose first token sits INSIDE the head render (the
/// user content ends with the end token, so the pair straddles the seam).
/// Only the byte-offset seam check can refuse this — every other guard
/// passes. Mutant receipt: seam check -> true derives Some(len-3) where
/// the reference tip is the head length.
#[test]
fn forced_suffix_straddling_the_seam_is_refused_by_the_offset_check_only() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_SPECIAL);
    let end = bundle.tokenizer().token_id("<|end|>").expect("end id");
    let hello = bundle.tokenizer().token_id("hello").expect("hello id");
    let messages = msgs(&[("user", "x<|end|>")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    let forced = [end, end, hello];
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary_forced(&messages, &[], Thinking::Default, &forced)
        .expect("derived");
    assert!(
        ids.ends_with(&forced),
        "fixture shape: the full ids must end with the forced suffix"
    );
    assert!(
        !exact,
        "a suffix whose first token starts inside the head must never derive"
    );
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip, "tip must equal the reference");
    assert_eq!(three.1.root, derived.root, "root must equal the reference");
}

/// Byte offsets, not char offsets: a multibyte emoji right before the
/// generation suffix. The seam check compares the candidate token's
/// offset against head.len() in BYTES ("user:hi" + 4-byte emoji); if a
/// tokenizers version ever reported char offsets, the comparison would
/// miss, derivation would silently fall back, and this test would fail.
/// It must derive.
#[test]
fn multibyte_seam_still_derives_with_byte_offsets() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_SPECIAL);
    let messages = msgs(&[("user", "hi🎉")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    assert!(
        exact,
        "the multibyte seam must still derive (offsets are bytes)"
    );
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip);
    assert_eq!(three.1.root, derived.root);
}

/// `rstrip=true` lets the matcher consume whitespace AFTER the token's
/// text, so the split is not byte-faithful. Load-time eligibility must
/// refuse; ids and boundaries still equal the three-pass.
#[test]
fn rstrip_tokenizer_never_derives() {
    let (_d, bundle) = fixture_bundle_with(TOKENIZER_JSON_RSTRIP, TEMPLATE_CONTENT);
    assert!(
        !bundle.tokenizer().boundary_eligible(),
        "an rstrip token disqualifies derivation"
    );
    let messages = msgs(&[("user", "hello")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    assert!(!exact, "an rstrip tokenizer must never derive");
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip);
    assert_eq!(three.1.root, derived.root);
}

/// A configured padding (Fixed) appends pad ids, so the id stream is not
/// the plain encoding. Load-time eligibility must refuse; ids and
/// boundaries still equal the three-pass.
#[test]
fn padded_tokenizer_never_derives() {
    let (_d, bundle) = fixture_bundle_with(TOKENIZER_JSON_PADDED, TEMPLATE_CONTENT);
    assert!(
        !bundle.tokenizer().boundary_eligible(),
        "a configured padding disqualifies derivation"
    );
    let messages = msgs(&[("user", "hello")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    assert!(!exact, "a padding tokenizer must never derive");
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip);
    assert_eq!(three.1.root, derived.root);
}

/// Work leg: on an ELIGIBLE tokenizer whose suffix never starts with an
/// added token (TEMPLATE_PLAIN), the pre-screen skips the suffix encode
/// and the fallback does no more work than three-pass.
#[test]
fn plain_suffix_eligible_fallback_does_no_more_work_than_three_pass() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_PLAIN);
    let messages = msgs(&[("user", "hello"), ("assistant", "world"), ("user", "hello")]);
    assert!(
        bundle.tokenizer().boundary_eligible(),
        "fixture must be eligible — this leg is the eligible fallback"
    );
    bundle.reset_counts();
    let reference = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("reference");
    let reference_counts = bundle.counts();
    bundle.reset_counts();
    let fell_back = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    let fallback_counts = bundle.counts();
    assert!(!fell_back.2, "plain suffix must fall back");
    assert_eq!(reference.1.tip, fell_back.1.tip);
    assert_eq!(reference.1.root, fell_back.1.root);
    assert!(
        fallback_counts.0 <= reference_counts.0 && fallback_counts.1 <= reference_counts.1,
        "eligible fallback: derived {fallback_counts:?} must not exceed reference {reference_counts:?}"
    );
}

/// Exact counterexample (Replace "ox" -> ""): the user turn ends 'o',
/// the generation prompt starts 'x<|end|>hello', so the deleted bytes
/// straddle the head/suffix seam. The suffix's emitted ids START with
/// the added end token (the dropped 'x' emits nothing), so an id-only
/// check derives Some(1) where the reference tip is None (the head
/// encode "helloo" -> [hello, o] is not a prefix of the full
/// "hello<|end|>hello" ids). The raw-text pre-screen refuses the suffix
/// ("x..." starts with no added text) and, with the screen bypassed, the
/// byte-0 offset guard is what refuses (the first token starts at suffix
/// byte 1, not 0).
#[test]
fn normalized_away_seam_bytes_never_derives() {
    let template =
        r#"{{ messages[0].content }}{% if add_generation_prompt %}x<|end|>hello{% endif %}"#;
    let (_d, bundle) = fixture_bundle_with(TOKENIZER_JSON_STRIP_OX, template);
    let messages = msgs(&[("user", "helloo")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    assert_eq!(three.1.tip, None, "reference: head ids are not a prefix");
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    assert!(
        !exact,
        "a suffix whose leading bytes normalize away must never derive"
    );
    assert_eq!(three.0, ids, "ids must equal the reference");
    assert_eq!(three.1.tip, derived.tip, "tip must equal the reference");
    assert_eq!(three.1.root, derived.root, "root must equal the reference");
}

/// `normalized=true` on the boundary token — normalized matching can
/// rewrite the token's text (case folding and the like), so the split is
/// not byte-faithful. Must fall back.
#[test]
fn normalized_added_token_never_derives() {
    let (_d, bundle) = fixture_bundle_with(TOKENIZER_JSON_NORMALIZED, TEMPLATE_CONTENT);
    assert!(
        !bundle.tokenizer().boundary_eligible(),
        "normalized flag disqualifies"
    );
    let messages = msgs(&[("user", "hello")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    assert!(!exact, "a normalized-matched token must never derive");
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip);
    assert_eq!(three.1.root, derived.root);
}

/// `single_word=true` on the boundary token — word-boundary matching is
/// contextual, so the seam is not inert. Must fall back (the `<|` start
/// does not rescue it: the flag itself is the hazard).
#[test]
fn single_word_added_token_never_derives() {
    let (_d, bundle) = fixture_bundle_with(TOKENIZER_JSON_SINGLE_WORD, TEMPLATE_CONTENT);
    assert!(
        !bundle.tokenizer().boundary_eligible(),
        "single_word flag disqualifies"
    );
    let messages = msgs(&[("user", "hello")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    assert!(!exact, "a single_word boundary token must never derive");
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip);
    assert_eq!(three.1.root, derived.root);
}

/// A single_word token in the vocabulary with a word-character suffix
/// start — the seam could feed a word-boundary match across the
/// head/suffix boundary. The seam guard must refuse (no derivation), even
/// though the boundary token itself matches cleanly.
#[test]
fn word_start_suffix_with_single_word_vocab_never_derives() {
    // Two added tokens: "hellox" clean (all flags false, word-char start —
    // the generation block emits it) and "mark" with single_word=true (so
    // the vocabulary is NOT single-word-free). Clean alone must not derive.
    let template = r#"{{ messages[0].content }}{% if add_generation_prompt %}hellox{% endif %}"#;
    let (_d, bundle) = fixture_bundle_with(TOKENIZER_JSON_WORD_START, template);
    let messages = msgs(&[("user", "hello")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    assert!(
        !exact,
        "a word-char suffix start with single_word tokens in the vocab must not derive"
    );
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip);
    assert_eq!(three.1.root, derived.root);
}

/// A single_word token IN THE HEAD — 'aa' matches at the head's end (word
/// boundary), but the full string's following word char 'b' invalidates
/// that match, so encode(full) != encode(head)++encode(suffix). The
/// load-time eligibility (any single_word token disqualifies) must refuse
/// the fast path: tip None, derived false.
#[test]
fn single_word_head_token_never_derives() {
    // Added 'aa' with single_word=true (id 14 in the fixture vocab), 'b'
    // clean reusing vocab id 10.
    let template = r#"{{ messages[0].content }}{% if add_generation_prompt %}bhello{% endif %}"#;
    let (_d, bundle) = fixture_bundle_with(TOKENIZER_JSON_HEAD_SINGLE_WORD, template);
    assert!(
        !bundle.tokenizer().boundary_eligible(),
        "a single_word token anywhere disqualifies derivation"
    );
    let messages = msgs(&[("user", "hello aa")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    // Head 'hello aa' encodes [8, 4, AA]; the full string's 'b' after 'aa'
    // breaks the single_word match, so the head ids are NOT a prefix.
    assert_eq!(three.1.tip, None, "reference: head ids are not a prefix");
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    assert!(!exact, "single_word in the head must never derive");
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip);
    assert_eq!(three.1.root, derived.root);
}

/// A configured LEFT truncation rewrites the id stream (max_length 4), so
/// suffix subtraction proves nothing. The load-time eligibility
/// (truncation configured disqualifies) must refuse.
#[test]
fn left_truncated_tokenizer_never_derives() {
    let (_d, bundle) = fixture_bundle_with(TOKENIZER_JSON_TRUNCATED, TEMPLATE_CONTENT);
    assert!(
        !bundle.tokenizer().boundary_eligible(),
        "a configured truncation disqualifies derivation"
    );
    let messages = msgs(&[("user", "hello hello hello")]);
    let three = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("three-pass");
    assert_eq!(
        three.1.tip, None,
        "reference: truncated encodes are not prefixes"
    );
    let (ids, derived, exact) = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    assert!(!exact, "a truncating tokenizer must never derive");
    assert_eq!(three.0, ids);
    assert_eq!(three.1.tip, derived.tip);
    assert_eq!(three.1.root, derived.root);
}

/// The derived path does no more bundle-level render and encode work than
/// the three-pass reference — on the deriving path AND the fallback path.
/// Counters are instrumented on every bundle-level render/encode call
/// site.
#[test]
fn derived_path_does_no_more_work_than_three_pass() {
    let (_d, bundle) = fixture_bundle(TEMPLATE_SPECIAL);
    let messages = msgs(&[("user", "hello world"), ("assistant", "hello")]);
    // Deriving path.
    bundle.reset_counts();
    let reference = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("reference");
    let reference_counts = bundle.counts();
    bundle.reset_counts();
    let derived = bundle
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    let derived_counts = bundle.counts();
    assert!(derived.2, "fixture derives on this path");
    assert_eq!(reference.1.tip, derived.1.tip);
    assert!(
        derived_counts.0 <= reference_counts.0 && derived_counts.1 <= reference_counts.1,
        "deriving path: derived {derived_counts:?} must not exceed reference {reference_counts:?}"
    );
    // Fallback path (ineligible tokenizer forces it).
    let (_d2, ineligible) = fixture_bundle_with(TOKENIZER_JSON_TRUNCATED_WIDE, TEMPLATE_SPECIAL);
    ineligible.reset_counts();
    let reference2 = ineligible
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("reference");
    let reference2_counts = ineligible.counts();
    ineligible.reset_counts();
    let fell_back = ineligible
        .encode_prompt_with_derived_boundary(&messages, &[], Thinking::Default)
        .expect("derived");
    let fallback_counts = ineligible.counts();
    assert!(!fell_back.2, "ineligible bundle falls back");
    assert_eq!(reference2.1.tip, fell_back.1.tip);
    assert!(
        fallback_counts.0 <= reference2_counts.0 && fallback_counts.1 <= reference2_counts.1,
        "fallback path: derived {fallback_counts:?} must not exceed reference {reference2_counts:?}"
    );

    // Alignment-rejected case (same counter test): a forced suffix the
    // full ids do NOT end with is rejected without extra encodes.
    let (_d3, eligible) = fixture_bundle(TEMPLATE_SPECIAL);
    // Starts with the added end id (so the override is honored; ids can
    // remap at load, so ask the tokenizer) but the block never ends
    // 'end,h,e' — the alignment guard rejects.
    let end = eligible.tokenizer().token_id("<|end|>").expect("end id");
    let stray = [end, 0, 1];
    eligible.reset_counts();
    let reference3 = eligible
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("reference");
    let reference3_counts = eligible.counts();
    eligible.reset_counts();
    let rejected = eligible
        .encode_prompt_with_derived_boundary_forced(&messages, &[], Thinking::Default, &stray)
        .expect("derived");
    let rejected_counts = eligible.counts();
    assert!(
        !rejected.2,
        "a stray forced suffix must be rejected (tail {:?})",
        &rejected.0[rejected.0.len().saturating_sub(4)..]
    );
    assert_eq!(reference3.1.tip, rejected.1.tip);
    assert!(
        rejected_counts.0 <= reference3_counts.0 && rejected_counts.1 <= reference3_counts.1,
        "alignment-rejected: derived {rejected_counts:?} must not exceed reference {reference3_counts:?}"
    );
}
