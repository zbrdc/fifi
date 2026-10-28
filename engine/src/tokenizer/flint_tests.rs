//! Portable tests against the complete publisher tokenizer, not a reduced stand-in.
use super::*;
use crate::exec_fake::FakeExecutor;
use crate::family::executor::{ExecutorOps, GenerationState};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::OnceLock;

const TEMPLATE: &str = include_str!("../../testdata/flint/tokenizer/chat_template.jinja");
const GOLDENS: &str = include_str!("../../testdata/flint/tokenizer/goldens.json");

fn hash(hex: &str) -> [u8; 32] {
    std::array::from_fn(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
}

pub(crate) fn bundle() -> &'static TokenizerBundle {
    static BUNDLE: OnceLock<TokenizerBundle> = OnceLock::new();
    BUNDLE.get_or_init(load_bundle)
}

pub(crate) fn fixture_dir() -> PathBuf {
    std::env::var_os("FLINT_MODEL_DIR")
        .map(PathBuf::from)
        .expect("set FLINT_MODEL_DIR to the pinned publisher model directory")
}

pub(crate) fn load_bundle() -> TokenizerBundle {
    TokenizerBundle::load(
        &fixture_dir(),
        &TokenizerPin {
            tokenizer_json: hash(
                "a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68",
            ),
            tokenizer_config_json: hash(
                "159ac88217e90ffd0a4a294726048bb302829e82e1958dc85bb203aa79138fbf",
            ),
            chat_template_jinja: hash(
                "7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466",
            ),
            generation_config_json: hash(
                "a499c71895e008abd572558a5ad5904feaed92cf2a3b6e2cb0e9cabf03b96ed3",
            ),
            vocab_size: 150_144,
        },
    )
    .expect("complete pinned Flint tokenizer fixture")
}

fn message(row: &Value) -> PromptMessage {
    PromptMessage {
        role: match row["role"].as_str().unwrap() {
            "system" => Role::System,
            "developer" => Role::Developer,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => panic!("unknown fixture role {other}"),
        },
        content: row["content"].as_str().map(str::to_owned),
        reasoning_content: row["reasoning_content"].as_str().map(str::to_owned),
        name: None,
        tool_call_id: row["tool_call_id"].as_str().map(str::to_owned),
        tool_calls: row["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|call| ToolCallSpec {
                id: call["id"].as_str().unwrap().to_owned(),
                name: call["name"].as_str().unwrap().to_owned(),
                arguments: call["arguments"].to_string(),
            })
            .collect(),
    }
}

#[test]
#[ignore = "requires FLINT_MODEL_DIR with the pinned publisher tokenizer"]
fn publisher_rendered_bytes_and_ids_match_for_all_histories() {
    let fixture: Value = serde_json::from_str(GOLDENS).unwrap();
    let template = ChatTemplate::from_str(TEMPLATE).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let messages: Vec<_> = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(message)
            .collect();
        let tools: Vec<_> = case["tools"]
            .as_array()
            .unwrap()
            .iter()
            .cloned()
            .map(ToolSpec)
            .collect();
        for reference in case["references"].as_array().unwrap() {
            let ids: Vec<u32> = serde_json::from_value(reference["ids"].clone()).unwrap();
            for thinking in [Thinking::Default, Thinking::On, Thinking::Off] {
                let rendered = if reference["generation_prompt"] == true {
                    assert_eq!(
                        bundle().encode_prompt(&messages, &tools, thinking).unwrap(),
                        ids
                    );
                    template.render(&messages, &tools, thinking).unwrap()
                } else {
                    template
                        .render_without_generation_prompt(&messages, &tools, thinking)
                        .unwrap()
                };
                assert_eq!(
                    rendered,
                    reference["rendered"].as_str().unwrap(),
                    "{}",
                    case["name"]
                );
                assert_eq!(
                    bundle().tokenizer().encode(&rendered).unwrap(),
                    ids,
                    "{}",
                    case["name"]
                );
                assert!(!ids.contains(&150_000), "no implicit BOS");
                assert!(!ids.contains(&150_001), "no implicit EOS");
            }
        }
    }
}

#[test]
#[ignore = "requires FLINT_MODEL_DIR with the pinned publisher tokenizer"]
fn publisher_tokenizer_structure_and_grammar_are_pinned() {
    bundle(); // Validate all four publisher hashes before the structural checks.
    let data: Value =
        serde_json::from_slice(&std::fs::read(fixture_dir().join("tokenizer.json")).unwrap())
            .unwrap();
    assert_eq!(data["model"]["type"], "BPE");
    assert_eq!(data["model"]["vocab"].as_object().unwrap().len(), 150_000);
    assert_eq!(data["model"]["merges"].as_array().unwrap().len(), 149_744);
    assert_eq!(data["added_tokens"].as_array().unwrap().len(), 144);
    assert!(data["normalizer"].is_null());
    assert!(data["post_processor"].is_null());
    let steps = data["pre_tokenizer"]["pretokenizers"].as_array().unwrap();
    assert_eq!(steps.len(), 4);
    assert!(
        steps[..3]
            .iter()
            .all(|step| step["type"] == "Split" && step["pattern"]["Regex"].is_string())
    );
    assert_eq!(steps[3]["type"], "ByteLevel");
    assert_eq!(bundle().vocab_size(), 150_144);
    assert_eq!(
        bundle().tokenizer().token_id("<|begin_of_text|>"),
        Some(150_000)
    );
    assert_eq!(bundle().tool_grammar(), toolcall::ToolGrammar::Flint);
    let other = ChatTemplate::from_str("{{ messages[0].content }}").unwrap();
    assert_eq!(other.grammar, toolcall::ToolGrammar::Qwen);
}

#[test]
fn legacy_templates_keep_developer_alias_and_ignore_reasoning_replay() {
    let template = ChatTemplate::from_str(
        "{{ messages[0].role }}|{{ messages[0].reasoning_content|default('absent') }}",
    )
    .unwrap();
    let message = PromptMessage {
        role: Role::Developer,
        content: Some("brief".into()),
        reasoning_content: Some("old reasoning".into()),
        name: None,
        tool_call_id: None,
        tool_calls: vec![],
    };
    assert_eq!(
        template.render(&[message], &[], Thinking::Default).unwrap(),
        "system|absent"
    );
}

#[test]
#[ignore = "requires FLINT_MODEL_DIR with the pinned publisher tokenizer"]
fn every_publisher_eos_finishes_incremental_decode_without_marker_leaks() {
    assert_eq!(
        bundle().eos_token_ids(),
        [150_001, 150_003, 150_005, 150_018]
    );
    let text = "héllo 🌍 <think>check</think><tool_call>run</tool_call>";
    let ids = bundle().tokenizer().encode(text).unwrap();
    for &eos in bundle().eos_token_ids() {
        let mut script: Vec<_> = ids.iter().map(|&id| vec![id]).collect();
        script.push(vec![eos]);
        let mut executor = FakeExecutor::new(script, bundle().eos_token_ids(), 1024);
        let mut session = Default::default();
        executor.boundary(&mut session);
        executor.prefill(&mut session, &[150_003]).unwrap();
        let mut state = GenerationState::new(IncrementalDecoder::new(bundle().tokenizer()));
        let mut assembler = OutputAssembler::new(vec![]).unwrap();
        let mut output = String::new();
        loop {
            let batch = executor.decode(&mut session, &mut state).unwrap();
            let delta = assembler.push_batch(&batch).unwrap();
            output.push_str(&delta.text);
            if let Some(finish) = delta.finish {
                assert_eq!(finish, DecodeFinish::Stop);
                break;
            }
        }
        assert_eq!(output, text, "EOS {eos}");
    }
}
