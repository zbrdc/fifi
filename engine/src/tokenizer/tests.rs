use super::{
    ChatTemplate, DecodeBatch, DecodeFinish, IncrementalDecoder, OutputAssembler, OutputDelta,
    PromptMessage, Role, StopHit, StopMatcher, Thinking, Tokenizer, TokenizerBundle, TokenizerPin,
    ToolCallSpec, ToolSpec,
};
use crate::EngError;
use crate::exec_fake::FakeExecutor;
use crate::family::executor::{ExecutorOps, GenerationState};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

const VOCAB_SIZE: u32 = 16;
const EOS_ID: u32 = 15;

const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":100,"content":" world","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":101,"content":"<|end|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"hello":8,"a":9,"b":10,"he":11,"hel":12,"hell":13},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;
const TOKENIZER_CONFIG_JSON: &str = r#"{"tokenizer_class":"TestTokenizer","eos_token":"<|end|>"}"#;
const GENERATION_CONFIG_JSON: &str =
    r#"{"bos_token_id":15,"eos_token_id":[15,11],"temperature":1.0}"#;
const SECOND_EOS_ID: u32 = 11;
const CHAT_TEMPLATE: &str = "{% if enable_thinking is defined %}a{% if enable_thinking %}b{% endif %}{% endif %}{% for t in tools %}{{ t.name }}{% endfor %}{{ messages[0].content }}";
const GENERATION_TEMPLATE: &str =
    "{{ 'GEN' if add_generation_prompt else 'NOGEN' }}|{{ messages[0].content }}";
const CONTEXT_TEMPLATE: &str = concat!(
    "{{ messages[0].role }}|{{ messages[0].content }}|{{ messages[0].name }}|",
    "{{ messages[0].tool_call_id }}|{{ messages[0].tool_calls | tojson }}|",
    "{{ tools | tojson }}|",
    "{% if enable_thinking is defined %}{{ enable_thinking }}{% else %}none{% endif %}",
);

fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

fn write_fixture(dir: &Path) -> TokenizerPin {
    fs::write(dir.join("tokenizer.json"), TOKENIZER_JSON).expect("write tokenizer.json");
    fs::write(dir.join("tokenizer_config.json"), TOKENIZER_CONFIG_JSON)
        .expect("write tokenizer_config.json");
    fs::write(dir.join("chat_template.jinja"), CHAT_TEMPLATE).expect("write chat_template.jinja");
    fs::write(dir.join("generation_config.json"), GENERATION_CONFIG_JSON)
        .expect("write generation_config.json");
    TokenizerPin {
        tokenizer_json: digest(TOKENIZER_JSON.as_bytes()),
        tokenizer_config_json: digest(TOKENIZER_CONFIG_JSON.as_bytes()),
        chat_template_jinja: digest(CHAT_TEMPLATE.as_bytes()),
        generation_config_json: digest(GENERATION_CONFIG_JSON.as_bytes()),
        vocab_size: VOCAB_SIZE,
    }
}

fn rewrite_generation_config(dir: &Path, pin: &mut TokenizerPin, text: &str) {
    fs::write(dir.join("generation_config.json"), text).expect("rewrite generation_config.json");
    pin.generation_config_json = digest(text.as_bytes());
}

fn rewrite_tokenizer_config(dir: &Path, pin: &mut TokenizerPin, text: &str) {
    fs::write(dir.join("tokenizer_config.json"), text).expect("rewrite tokenizer_config.json");
    pin.tokenizer_config_json = digest(text.as_bytes());
}

fn eos_ids_for(generation_config: &str) -> Vec<u32> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut pin = write_fixture(dir.path());
    rewrite_generation_config(dir.path(), &mut pin, generation_config);
    TokenizerBundle::load(dir.path(), &pin)
        .expect("load")
        .eos_token_ids()
        .to_vec()
}

fn checkpoint_message(result: Result<TokenizerBundle, EngError>) -> String {
    match result {
        Err(EngError::Checkpoint(message)) => message,
        other => panic!("expected Checkpoint, got {:?}", other.map(|_| "loaded")),
    }
}

#[test]
fn missing_tokenizer_json_fails_checkpoint() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pin = write_fixture(dir.path());
    fs::remove_file(dir.path().join("tokenizer.json")).expect("remove tokenizer.json");
    let message = checkpoint_message(TokenizerBundle::load(dir.path(), &pin));
    assert!(message.contains("tokenizer.json"), "message: {message}");
}

#[test]
fn template_hash_mismatch_fails_checkpoint() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut pin = write_fixture(dir.path());
    pin.chat_template_jinja[0] ^= 1;
    let message = checkpoint_message(TokenizerBundle::load(dir.path(), &pin));
    assert!(
        message.contains("chat_template.jinja"),
        "message: {message}"
    );
}

#[test]
fn load_reports_fixture_vocab_and_eos() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pin = write_fixture(dir.path());
    let bundle = TokenizerBundle::load(dir.path(), &pin).expect("load");
    assert_eq!(bundle.vocab_size(), VOCAB_SIZE);
    assert_eq!(bundle.eos_token_id(), EOS_ID);
    assert_eq!(bundle.identity().tokenizer_json, pin.tokenizer_json);
    assert_eq!(
        bundle.identity().generation_config_json,
        pin.generation_config_json
    );
    assert_eq!(
        bundle.identity().tokenizer_config_json,
        pin.tokenizer_config_json
    );
    assert_eq!(
        bundle.identity().chat_template_jinja,
        pin.chat_template_jinja
    );
}

#[test]
fn vocab_mismatch_fails_checkpoint() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut pin = write_fixture(dir.path());
    pin.vocab_size = VOCAB_SIZE - 1;
    let message = checkpoint_message(TokenizerBundle::load(dir.path(), &pin));
    assert!(message.contains("vocab"), "message: {message}");
}

#[test]
fn missing_eos_token_fails_checkpoint() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut pin = write_fixture(dir.path());
    fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"tokenizer_class":"TestTokenizer"}"#,
    )
    .expect("rewrite tokenizer_config.json");
    pin.tokenizer_config_json = digest(br#"{"tokenizer_class":"TestTokenizer"}"#);
    let message = checkpoint_message(TokenizerBundle::load(dir.path(), &pin));
    assert!(message.contains("eos"), "message: {message}");
}

#[test]
fn eos_token_outside_vocabulary_fails_checkpoint() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut pin = write_fixture(dir.path());
    rewrite_tokenizer_config(
        dir.path(),
        &mut pin,
        r#"{"tokenizer_class":"TestTokenizer","eos_token":"<|missing|>"}"#,
    );
    let message = checkpoint_message(TokenizerBundle::load(dir.path(), &pin));
    assert_eq!(
        message,
        "eos_token <|missing|> is not in the tokenizer vocabulary"
    );
}

fn fixture_tokenizer() -> Tokenizer {
    Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).expect("parse fixture tokenizer")
}

fn user_message(content: &str) -> PromptMessage {
    PromptMessage {
        role: Role::User,
        content: Some(content.to_string()),
        name: None,
        tool_call_id: None,
        tool_calls: vec![],
    }
}

fn example_tool() -> ToolSpec {
    ToolSpec(serde_json::json!({
        "name": "hello",
        "description": "d",
        "parameters": {"type": "object"},
    }))
}

#[test]
fn wrapper_encode_never_adds_special_tokens() {
    let tokenizer = fixture_tokenizer();
    assert_eq!(
        tokenizer.encode("hello world").expect("encode"),
        vec![8, 14]
    );
}

#[test]
fn wrapper_decode_skips_special_tokens() {
    let tokenizer = fixture_tokenizer();
    assert_eq!(tokenizer.decode(&[8, 14]).expect("decode"), "hello world");
    assert_eq!(tokenizer.decode(&[15]).expect("decode"), "");
}

#[test]
fn wrapper_decode_rejects_ids_outside_the_vocab() {
    let tokenizer = fixture_tokenizer();
    let error = tokenizer.decode(&[999]).expect_err("outside vocab");
    match error {
        EngError::Checkpoint(message) => {
            assert!(message.contains("outside"), "message: {message}")
        }
        other => panic!("expected Checkpoint, got {other:?}"),
    }
}

#[test]
fn render_pins_the_message_tool_and_thinking_context() {
    let template = ChatTemplate::from_str(CONTEXT_TEMPLATE).expect("template");
    let message = PromptMessage {
        role: Role::User,
        content: Some("hi".to_string()),
        name: Some("n".to_string()),
        tool_call_id: Some("tid".to_string()),
        tool_calls: vec![ToolCallSpec {
            id: "call-1".to_string(),
            name: "fn".to_string(),
            arguments: "{}".to_string(),
        }],
    };
    let tools = vec![ToolSpec(serde_json::json!({
        "name": "lookup",
        "description": "d",
        "parameters": {"type": "object"},
    }))];
    let rendered = template
        .render(&[message], &tools, Thinking::On)
        .expect("render");
    let mut parts = rendered.splitn(7, '|');
    assert_eq!(parts.next(), Some("user"));
    assert_eq!(parts.next(), Some("hi"));
    assert_eq!(parts.next(), Some("n"));
    assert_eq!(parts.next(), Some("tid"));
    let calls: serde_json::Value =
        serde_json::from_str(parts.next().expect("tool_calls")).expect("tool_calls json");
    assert_eq!(calls[0]["id"], "call-1");
    assert_eq!(calls[0]["name"], "fn");
    assert_eq!(calls[0]["arguments"], serde_json::json!({}));
    let tools_json: serde_json::Value =
        serde_json::from_str(parts.next().expect("tools")).expect("tools json");
    assert_eq!(tools_json[0]["name"], "lookup");
    assert_eq!(tools_json[0]["description"], "d");
    assert_eq!(tools_json[0]["parameters"]["type"], "object");
    assert_eq!(parts.next(), Some("True"));
}

#[test]
fn render_rejects_non_object_tool_call_arguments() {
    let template = ChatTemplate::from_str(CONTEXT_TEMPLATE).expect("template");
    for arguments in ["[1]", "\"x\""] {
        let message = PromptMessage {
            role: Role::Assistant,
            content: None,
            name: None,
            tool_call_id: None,
            tool_calls: vec![ToolCallSpec {
                id: "call-1".to_string(),
                name: "fn".to_string(),
                arguments: arguments.to_string(),
            }],
        };
        match template.render(&[message], &[], Thinking::Default) {
            Err(EngError::BadRequest(text)) => {
                assert_eq!(text, "tool call arguments must be a JSON object");
            }
            other => panic!("arguments {arguments}: expected BadRequest, got {other:?}"),
        }
    }
}

#[test]
fn thinking_default_omits_the_flag() {
    let template = ChatTemplate::from_str(CONTEXT_TEMPLATE).expect("template");
    let rendered = template
        .render(&[user_message("hi")], &[], Thinking::Default)
        .expect("render");
    assert!(rendered.ends_with("none"), "rendered: {rendered}");
    let rendered = template
        .render(&[user_message("hi")], &[], Thinking::Off)
        .expect("render");
    assert!(rendered.ends_with("False"), "rendered: {rendered}");
}

#[test]
fn render_sets_the_generation_prompt_flag() {
    let template = ChatTemplate::from_str(GENERATION_TEMPLATE).expect("template");
    let rendered = template
        .render(&[user_message("hi")], &[], Thinking::Default)
        .expect("render");
    assert_eq!(rendered, "GEN|hi");
}

#[test]
fn encode_prompt_encodes_the_rendered_prompt() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pin = write_fixture(dir.path());
    let bundle = TokenizerBundle::load(dir.path(), &pin).expect("load");
    let ids = bundle
        .encode_prompt(&[user_message("hello world")], &[], Thinking::Default)
        .expect("encode");
    assert_eq!(ids, vec![8, 14]);
}

#[test]
fn encode_prompt_applies_thinking_and_tools() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pin = write_fixture(dir.path());
    let bundle = TokenizerBundle::load(dir.path(), &pin).expect("load");
    let ids = bundle
        .encode_prompt(&[user_message(" world")], &[example_tool()], Thinking::On)
        .expect("encode");
    assert_eq!(ids, vec![9, 10, 8, 14]);
    let ids = bundle
        .encode_prompt(&[user_message("hello world")], &[], Thinking::Off)
        .expect("encode");
    assert_eq!(ids, vec![9, 8, 14]);
}

#[test]
fn empty_message_list_fails_bad_request() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pin = write_fixture(dir.path());
    let bundle = TokenizerBundle::load(dir.path(), &pin).expect("load");
    let error = bundle
        .encode_prompt(&[], &[], Thinking::Default)
        .expect_err("empty list");
    assert!(matches!(error, EngError::BadRequest(_)), "error: {error:?}");
}

#[test]
fn empty_rendered_prompt_fails_bad_request() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pin = write_fixture(dir.path());
    let bundle = TokenizerBundle::load(dir.path(), &pin).expect("load");
    let error = bundle
        .encode_prompt(&[user_message("")], &[], Thinking::Default)
        .expect_err("empty render");
    assert!(matches!(error, EngError::BadRequest(_)), "error: {error:?}");
}

const SPLIT_TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":4,"content":"<end>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"Ã":0,"©":1,"Ã©":2,"a":3},"merges":[]}}"#;

fn split_tokenizer() -> Tokenizer {
    Tokenizer::from_bytes(SPLIT_TOKENIZER_JSON.as_bytes()).expect("parse split fixture tokenizer")
}

#[test]
fn incremental_decoder_holds_partial_utf8_until_the_sequence_completes() {
    let mut decoder = IncrementalDecoder::new(&split_tokenizer());
    assert_eq!(decoder.push(0).expect("first byte"), "");
    assert_eq!(decoder.push(1).expect("second byte"), "\u{e9}");
}

#[test]
fn incremental_decoder_flushes_the_incomplete_tail() {
    let mut decoder = IncrementalDecoder::new(&split_tokenizer());
    assert_eq!(decoder.push(0).expect("first byte"), "");
    assert_eq!(decoder.finish().expect("finish"), "\u{fffd}");
}

#[test]
fn incremental_decoder_flush_is_empty_after_complete_text() {
    let mut decoder = IncrementalDecoder::new(&split_tokenizer());
    assert_eq!(decoder.push(3).expect("letter"), "a");
    assert_eq!(decoder.finish().expect("finish"), "");
}

#[test]
fn incremental_decoder_skips_special_tokens() {
    let mut decoder = IncrementalDecoder::new(&split_tokenizer());
    assert_eq!(decoder.push(4).expect("special token"), "");
    assert_eq!(decoder.finish().expect("finish"), "");
}

#[test]
fn incremental_decoder_rejects_ids_outside_the_vocab() {
    let mut decoder = IncrementalDecoder::new(&split_tokenizer());
    let error = decoder.push(999).expect_err("out of range id");
    assert!(
        matches!(&error, EngError::Checkpoint(message) if message.contains("outside")),
        "error: {error:?}"
    );
}

fn stop(stops: &[&str]) -> StopMatcher {
    StopMatcher::new(stops.iter().map(|text| (*text).to_string()).collect())
        .expect("build stop matcher")
}

#[test]
fn stop_matches_across_pushes() {
    let mut matcher = stop(&["stop"]);
    assert_eq!(matcher.push_str("st"), None);
    assert_eq!(matcher.push_str("o"), None);
    assert_eq!(matcher.push_str("pX"), Some(StopHit { stop: 0, end: 4 }));
}

#[test]
fn stop_keeps_the_tail_without_a_false_prefix() {
    let mut matcher = stop(&["stop"]);
    assert_eq!(matcher.push_str("sto"), None);
    assert_eq!(matcher.push_str("p"), Some(StopHit { stop: 0, end: 4 }));
}

#[test]
fn stop_tie_resolves_to_the_lowest_index() {
    let mut matcher = stop(&["ab", "b"]);
    assert_eq!(matcher.push_str("aab"), Some(StopHit { stop: 0, end: 3 }));
}

#[test]
fn stop_earliest_end_wins_across_lengths() {
    let mut matcher = stop(&["xy", "yz"]);
    assert_eq!(matcher.push_str("xyz"), Some(StopHit { stop: 0, end: 2 }));
}

#[test]
fn empty_stop_list_never_matches() {
    let mut matcher = stop(&[]);
    assert_eq!(matcher.push_str("stop stop stop"), None);
    assert_eq!(matcher.push_str("stop"), None);
}

#[test]
fn stop_tail_stays_bounded_without_losing_matches() {
    let mut matcher = stop(&["abcdef"]);
    assert_eq!(matcher.push_str("abcde"), None);
    assert_eq!(matcher.push_str("xyz"), None);
    assert_eq!(matcher.push_str("f"), None);
    assert_eq!(
        matcher.push_str("abcdef"),
        Some(StopHit { stop: 0, end: 15 })
    );
}

#[test]
fn stop_matches_multi_byte_text_byte_exact() {
    let mut matcher = stop(&["\u{e9}!"]);
    assert_eq!(matcher.push_str("\u{e9}"), None);
    assert_eq!(matcher.push_str("!"), Some(StopHit { stop: 0, end: 3 }));
    let mut wrong_case = stop(&["\u{e9}!"]);
    assert_eq!(wrong_case.push_str("\u{c9}!"), None);
}

#[test]
fn empty_stop_string_fails_bad_request() {
    let error = StopMatcher::new(vec![String::new()]).expect_err("empty stop");
    assert!(matches!(error, EngError::BadRequest(_)), "error: {error:?}");
}

const OUTPUT_TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":7,"content":"<end>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"st":1,"o":2,"pX":3,"Ã":4,"©":5,"!":6},"merges":[]}}"#;

fn output_tokenizer() -> Tokenizer {
    Tokenizer::from_bytes(OUTPUT_TOKENIZER_JSON.as_bytes()).expect("parse output fixture tokenizer")
}

fn new_assembler(stops: &[&str]) -> OutputAssembler {
    OutputAssembler::new(stops.iter().map(|text| (*text).to_string()).collect())
        .expect("build output assembler")
}

fn delta(text: &str, finish: Option<DecodeFinish>) -> OutputDelta {
    OutputDelta {
        text: text.to_string(),
        finish,
    }
}

fn batch(tokens: &[u32], text: &str, finish: Option<DecodeFinish>) -> DecodeBatch {
    DecodeBatch {
        tokens: tokens.to_vec(),
        text: text.to_string(),
        finish,
    }
}

struct Job {
    executor: FakeExecutor,
    session: <FakeExecutor as ExecutorOps>::Session,
    generation: GenerationState,
    assembler: OutputAssembler,
}

// Runs one job through the fake executor, as the worker does: prefill, then
// `decode` until a batch has a finish, feeding each batch to the assembler.
fn job(stops: &[&str], max_tokens: u32, eos: &[u32], script: Vec<Vec<u32>>) -> Job {
    let mut executor = FakeExecutor::new(script, eos, max_tokens);
    let mut session = Default::default();
    executor.boundary(&mut session);
    executor.prefill(&mut session, &[1]).expect("prefill");
    Job {
        executor,
        session,
        generation: GenerationState::new(IncrementalDecoder::new(&output_tokenizer())),
        assembler: new_assembler(stops),
    }
}

impl Job {
    fn step(&mut self) -> (DecodeBatch, OutputDelta) {
        let batch = self
            .executor
            .decode(&mut self.session, &mut self.generation)
            .expect("decode");
        let out = self.assembler.push_batch(&batch).expect("push batch");
        (batch, out)
    }

    fn run(&mut self) -> Vec<OutputDelta> {
        let mut out = Vec::new();
        loop {
            let (batch, delta) = self.step();
            out.push(delta);
            if batch.finish.is_some() || out.last().is_some_and(|d| d.finish.is_some()) {
                return out;
            }
        }
    }
}

fn run_ids(stops: &[&str], max_tokens: u32, eos: &[u32], ids: &[u32]) -> Vec<OutputDelta> {
    job(
        stops,
        max_tokens,
        eos,
        ids.iter().map(|&id| vec![id]).collect(),
    )
    .run()
}

const STOP_FINISH: Option<DecodeFinish> = Some(DecodeFinish::Stop);

#[test]
fn output_assembler_emits_plain_text_deltas() {
    let mut j = job(&["stop"], 5, &[OUTPUT_EOS], vec![vec![0]]);
    assert_eq!(j.step().1, delta("a", None));
}

#[test]
fn output_assembler_consumes_eos_without_text() {
    assert_eq!(
        run_ids(&["stop"], 5, &[OUTPUT_EOS], &[OUTPUT_EOS]),
        vec![delta("", STOP_FINISH)]
    );
}

#[test]
fn output_assembler_hides_the_stop_text() {
    assert_eq!(
        run_ids(&["stop"], 5, &[OUTPUT_EOS], &[0, 1, 2, 3]),
        vec![
            delta("a", None),
            delta("", None),
            delta("", None),
            delta("", STOP_FINISH),
        ]
    );
}

#[test]
fn output_assembler_marks_the_output_limit() {
    assert_eq!(
        run_ids(&["stop"], 1, &[OUTPUT_EOS], &[0]),
        vec![delta("a", Some(DecodeFinish::Length))]
    );
}

#[test]
fn output_assembler_waits_for_multi_byte_stop_completion() {
    assert_eq!(
        run_ids(&["\u{e9}!"], 5, &[OUTPUT_EOS], &[4, 5, 6]),
        vec![delta("", None), delta("", None), delta("", STOP_FINISH)]
    );
}

#[test]
fn output_assembler_returns_the_final_delta_for_every_later_batch() {
    let mut assembler = new_assembler(&["stop"]);
    let last = assembler
        .push_batch(&batch(&[0], "a", Some(DecodeFinish::Length)))
        .expect("push");
    assert_eq!(last, delta("a", Some(DecodeFinish::Length)));
    for _ in 0..7 {
        assert_eq!(
            assembler
                .push_batch(&batch(&[0], "a", None))
                .expect("push after finish"),
            delta("", Some(DecodeFinish::Length))
        );
    }
}

#[test]
fn fake_decode_counts_a_large_output_limit_without_overflow() {
    let mut j = job(&["stop"], u32::MAX, &[OUTPUT_EOS], vec![vec![0]; 49]);
    for _ in 0..49 {
        assert_eq!(j.step().1, delta("a", None));
    }
}

#[test]
fn a_zero_output_limit_finishes_with_length_and_no_tokens() {
    let mut j = job(&["stop"], 0, &[OUTPUT_EOS], vec![]);
    let (b, d) = j.step();
    assert_eq!(b, batch(&[], "", Some(DecodeFinish::Length)));
    assert_eq!(d, delta("", Some(DecodeFinish::Length)));
}

#[test]
fn output_assembler_releases_a_held_stop_prefix_at_eos() {
    assert_eq!(
        run_ids(&["stop"], 5, &[OUTPUT_EOS], &[0, 1, OUTPUT_EOS]),
        vec![delta("a", None), delta("", None), delta("st", STOP_FINISH)]
    );
}

#[test]
fn output_assembler_releases_a_held_stop_prefix_at_the_output_limit() {
    assert_eq!(
        run_ids(&["stop"], 2, &[OUTPUT_EOS], &[0, 1]),
        vec![delta("a", None), delta("st", Some(DecodeFinish::Length))]
    );
}

#[test]
fn output_assembler_flushes_a_pending_multibyte_tail_at_eos() {
    assert_eq!(
        run_ids(&["stop"], 5, &[OUTPUT_EOS], &[0, 4, OUTPUT_EOS]),
        vec![
            delta("a", None),
            delta("", None),
            delta("\u{fffd}", STOP_FINISH)
        ]
    );
}

#[test]
fn output_assembler_flushes_a_pending_multibyte_tail_at_the_output_limit() {
    assert_eq!(
        run_ids(&["stop"], 2, &[OUTPUT_EOS], &[0, 4]),
        vec![
            delta("a", None),
            delta("\u{fffd}", Some(DecodeFinish::Length))
        ]
    );
}

#[test]
fn output_assembler_checks_stops_against_the_flushed_tail() {
    assert_eq!(
        run_ids(&["a\u{fffd}"], 5, &[OUTPUT_EOS], &[0, 4, OUTPUT_EOS]),
        vec![delta("", None), delta("", None), delta("", STOP_FINISH)]
    );
    assert_eq!(
        run_ids(&["a\u{fffd}"], 2, &[OUTPUT_EOS], &[0, 4]),
        vec![delta("", None), delta("", STOP_FINISH)]
    );
}

#[test]
fn a_stop_completed_by_the_last_allowed_token_reports_stop() {
    assert_eq!(
        run_ids(&["stop"], 4, &[OUTPUT_EOS], &[0, 1, 2, 3]),
        vec![
            delta("a", None),
            delta("", None),
            delta("", None),
            delta("", STOP_FINISH),
        ]
    );
}

#[test]
fn output_assembler_finish_releases_held_text_for_a_cancelled_job() {
    let mut assembler = new_assembler(&["stop"]);
    assert_eq!(
        assembler
            .push_batch(&batch(&[0, 1], "ast", None))
            .expect("push"),
        delta("a", None)
    );
    assert_eq!(
        assembler.finish(DecodeFinish::Cancelled).expect("finish"),
        delta("st", Some(DecodeFinish::Cancelled))
    );
    assert_eq!(
        assembler
            .push_batch(&batch(&[0], "a", None))
            .expect("push after finish"),
        delta("", Some(DecodeFinish::Cancelled))
    );
    assert_eq!(
        assembler
            .finish(DecodeFinish::Length)
            .expect("second finish"),
        delta("", Some(DecodeFinish::Cancelled))
    );
}

#[test]
fn a_cancelled_batch_flushes_a_pending_multibyte_tail() {
    let mut j = job(&["stop"], 5, &[OUTPUT_EOS], vec![vec![0, 4]]);
    assert_eq!(j.step().1, delta("a", None));
    j.generation
        .cancel
        .store(true, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(j.step().1, delta("\u{fffd}", Some(DecodeFinish::Cancelled)));
}

#[test]
fn a_cancelled_batch_keeps_its_reason_and_hides_a_completed_stop() {
    let mut j = job(&["a\u{fffd}"], 5, &[OUTPUT_EOS], vec![vec![0, 4]]);
    assert_eq!(j.step().1, delta("", None));
    j.generation
        .cancel
        .store(true, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(j.step().1, delta("", Some(DecodeFinish::Cancelled)));
}

#[test]
fn output_assembler_finish_keeps_the_stop_reason_after_a_stop_hit() {
    let mut assembler = new_assembler(&["stop"]);
    assembler
        .push_batch(&batch(&[0, 1, 2, 3], "astopX", None))
        .expect("push");
    assert_eq!(
        assembler.finish(DecodeFinish::Cancelled).expect("finish"),
        delta("", STOP_FINISH)
    );
}

#[test]
fn output_assembler_stops_on_any_eos_id() {
    assert_eq!(
        run_ids(&["stop"], 5, &[OUTPUT_EOS, 6], &[0, 1, 6]),
        vec![delta("a", None), delta("", None), delta("st", STOP_FINISH)]
    );
    assert_eq!(
        run_ids(&["stop"], 5, &[OUTPUT_EOS, 6], &[0, OUTPUT_EOS]),
        vec![delta("a", None), delta("", STOP_FINISH)]
    );
}

// S8-T1: a scripted executor yields the expected delta texts and finish.
#[test]
fn s8_t1_scripted_executor_yields_deltas_and_finish() {
    let mut j = job(
        &["stop"],
        9,
        &[OUTPUT_EOS],
        vec![vec![0, 0], vec![0, OUTPUT_EOS]],
    );
    let (b1, d1) = j.step();
    assert_eq!((b1, d1), (batch(&[0, 0], "aa", None), delta("aa", None)));
    let (b2, d2) = j.step();
    assert_eq!(
        (b2, d2),
        (
            batch(&[0, OUTPUT_EOS], "a", STOP_FINISH),
            delta("a", STOP_FINISH)
        )
    );
    // The EOS id belongs to the consumed prefix and to the session history.
    assert_eq!(j.executor.history(&j.session), &[1, 0, 0, 0, OUTPUT_EOS]);
}

// S8-T2: the stop text stays in the batch text and tokens. The assembler hides it.
// The tokens after the stop stay committed until Phase 2 defines the discard rule
// (`family-descriptor.md`, consumed prefix).
#[test]
fn s8_t2_batch_keeps_the_stop_text_and_the_assembler_hides_it() {
    let mut j = job(&["stop"], 9, &[OUTPUT_EOS], vec![vec![0, 1, 2, 3, 0]]);
    let (b, d) = j.step();
    assert_eq!(b.tokens, vec![0, 1, 2, 3, 0]);
    assert_eq!(b.text, "astopXa");
    assert_eq!(b.finish, None);
    assert_eq!(d, delta("a", STOP_FINISH));
}

// S8-T3: batch text minus the stop text equals the concatenated delta texts.
#[test]
fn s8_t3_batch_text_minus_the_stop_equals_the_delta_texts() {
    let mut j = job(
        &["stop"],
        9,
        &[OUTPUT_EOS],
        vec![vec![0, 0], vec![1], vec![2, 3, 0]],
    );
    let mut batch_text = String::new();
    let mut delta_text = String::new();
    loop {
        let (b, d) = j.step();
        batch_text.push_str(&b.text);
        delta_text.push_str(&d.text);
        if d.finish.is_some() {
            break;
        }
    }
    assert_eq!(batch_text, "aastopXa");
    let before_stop = batch_text
        .split("stop")
        .next()
        .expect("text before the stop");
    assert_eq!(before_stop, delta_text);
    assert_eq!(delta_text, "aa");
}

#[test]
fn the_fake_enforces_the_worker_call_order() {
    let mut executor = FakeExecutor::new(vec![vec![0], vec![0]], &[OUTPUT_EOS], 1);
    let mut session = Default::default();
    let mut generation = GenerationState::new(IncrementalDecoder::new(&output_tokenizer()));
    executor
        .decode(&mut session, &mut generation)
        .expect_err("decode before prefill");
    executor
        .prefill(&mut session, &[])
        .expect_err("empty chunk");
    executor.prefill(&mut session, &[1]).expect("prefill");
    assert_eq!(executor.history(&session), &[1]);
    executor
        .decode(&mut session, &mut generation)
        .expect("decode");
    executor
        .decode(&mut session, &mut generation)
        .expect_err("decode after finish");
    executor.reset(&mut session).expect("reset");
    assert!(executor.history(&session).is_empty());
}
const OUTPUT_EOS: u32 = 7;

#[test]
fn wrapper_round_trip_restores_the_text() {
    let tokenizer = fixture_tokenizer();
    for text in ["hello world", "hello", "", "  ", "ab", "hello hello world"] {
        let ids = tokenizer.encode(text).expect("encode");
        assert_eq!(
            tokenizer.decode(&ids).expect("decode"),
            text,
            "text: {text:?}"
        );
    }
    let multibyte = Tokenizer::from_bytes(OUTPUT_TOKENIZER_JSON.as_bytes()).expect("fixture");
    for text in ["\u{e9}", "a\u{e9}!", "!!a"] {
        let ids = multibyte.encode(text).expect("encode");
        assert_eq!(
            multibyte.decode(&ids).expect("decode"),
            text,
            "text: {text:?}"
        );
    }
}

#[test]
fn template_raise_exception_maps_to_bad_request() {
    let template = ChatTemplate::from_str(
        "{% if messages[0].tool_call_id == 'missing' %}{{ raise_exception('tool call unknown') }}{% endif %}{{ messages[0].content }}",
    )
    .expect("parse template");
    let message = PromptMessage {
        role: Role::Tool,
        content: Some("result".to_string()),
        name: None,
        tool_call_id: Some("missing".to_string()),
        tool_calls: Vec::new(),
    };
    let error = template
        .render(&[message], &[], Thinking::Default)
        .expect_err("render");
    assert!(matches!(error, EngError::BadRequest(_)), "error: {error:?}");
    let text = format!("{error:?}");
    assert!(text.contains("tool call unknown"), "error: {text}");
}

#[test]
fn template_syntax_error_fails_checkpoint() {
    let error = ChatTemplate::from_str("{% if %}")
        .map(|_| ())
        .expect_err("syntax");
    assert!(matches!(error, EngError::Checkpoint(_)), "error: {error:?}");
}

#[test]
fn rendering_is_deterministic() {
    let template = ChatTemplate::from_str("{% for m in messages %}{{ m.content }}{% endfor %}")
        .expect("parse template");
    let messages = [user_message("hello"), user_message(" world")];
    let first = template
        .render(&messages, &[], Thinking::Default)
        .expect("render");
    let second = template
        .render(&messages, &[], Thinking::Default)
        .expect("render");
    assert_eq!(first, second);
    assert_eq!(first, "hello world");
}

#[test]
fn encode_prompt_prefix_stability_across_added_messages() {
    let tokenizer = Tokenizer::from_bytes(OUTPUT_TOKENIZER_JSON.as_bytes()).expect("fixture");
    let template = ChatTemplate::from_str("{% for m in messages %}{{ m.content }}{% endfor %}")
        .expect("parse template");
    let contents = ["a", "o", "!", "a"];
    let mut previous: Vec<u32> = Vec::new();
    for count in 1..=contents.len() {
        let messages: Vec<PromptMessage> =
            contents[..count].iter().map(|c| user_message(c)).collect();
        let prompt = template
            .render(&messages, &[], Thinking::Default)
            .expect("render");
        let ids = tokenizer.encode(&prompt).expect("encode");
        assert!(ids.starts_with(&previous), "count {count}");
        previous = ids;
    }
    assert_eq!(previous, vec![0, 2, 6, 0]);
}

const GOLDENS_JSON: &str = include_str!("../../testdata/tokenizer-goldens.json");
const CHAT_TEMPLATE_JINJA: &[u8] = include_bytes!("../../testdata/chat_template.jinja");
const CHAT_TEMPLATE_SHA256: &str =
    "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259";
const SYSTEM_MARKER: &str = "\u{3c}|im_start|\u{3e}system\n";
const GENERATION_MARKER: &str = "\u{3c}|im_start|\u{3e}assistant\n";
const THINK_PREAMBLE: &str = "\u{3c}think\u{3e}\n";
const EMPTY_THINK_BLOCK: &str = "\u{3c}think\u{3e}\n\n\u{3c}/think\u{3e}\n\n";

fn pinned_template() -> ChatTemplate {
    ChatTemplate::from_str(include_str!("../../testdata/chat_template.jinja"))
        .expect("pinned template")
}

fn goldens() -> serde_json::Value {
    serde_json::from_str(GOLDENS_JSON).expect("goldens JSON parses")
}

fn golden_renders(goldens: &serde_json::Value) -> &Vec<serde_json::Value> {
    assert_eq!(goldens["schema"], "p1-tokenizer-goldens-1");
    let renders = goldens["renders"].as_array().expect("renders array");
    assert!(!renders.is_empty(), "goldens need render rows");
    renders
}

fn golden_messages(json: &str) -> Vec<PromptMessage> {
    let rows: Vec<serde_json::Value> = serde_json::from_str(json).expect("messages JSON");
    rows.iter().map(golden_message).collect()
}

fn golden_message(row: &serde_json::Value) -> PromptMessage {
    let role = match row["role"].as_str().expect("role") {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        other => panic!("unknown golden role {other}"),
    };
    PromptMessage {
        role,
        content: optional_string(&row["content"]),
        name: optional_string(&row["name"]),
        tool_call_id: optional_string(&row["tool_call_id"]),
        tool_calls: golden_tool_calls(&row["tool_calls"]),
    }
}

fn optional_string(row: &serde_json::Value) -> Option<String> {
    row.as_str().map(str::to_string)
}

fn golden_tool_calls(row: &serde_json::Value) -> Vec<ToolCallSpec> {
    let Some(rows) = row.as_array() else {
        return Vec::new();
    };
    rows.iter()
        .map(|item| ToolCallSpec {
            id: item["id"].as_str().expect("id").to_string(),
            name: item["name"].as_str().expect("name").to_string(),
            arguments: golden_arguments(&item["arguments"]),
        })
        .collect()
}

fn golden_arguments(row: &serde_json::Value) -> String {
    match row {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Object(_) => row.to_string(),
        other => panic!("unsupported golden arguments {other:?}"),
    }
}

fn golden_tools(json: &str) -> Vec<ToolSpec> {
    let rows: Vec<serde_json::Value> = serde_json::from_str(json).expect("tools JSON");
    rows.iter().map(|item| ToolSpec(item.clone())).collect()
}

fn golden_thinking(row: &serde_json::Value) -> Thinking {
    match row["thinking"].as_str().expect("thinking") {
        "default" => Thinking::Default,
        "on" => Thinking::On,
        "off" => Thinking::Off,
        other => panic!("unknown golden thinking {other}"),
    }
}

#[test]
fn testdata_chat_template_matches_the_pinned_hash() {
    assert_eq!(CHAT_TEMPLATE_JINJA.len(), 7_764);
    let digest = digest(CHAT_TEMPLATE_JINJA);
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(hex, CHAT_TEMPLATE_SHA256);
}

#[test]
fn golden_renders_match_the_engine_render() {
    let goldens = goldens();
    let template = pinned_template();
    for row in golden_renders(&goldens) {
        let id = row["id"].as_str().expect("id");
        let messages = golden_messages(row["messages_json"].as_str().expect("messages_json"));
        let tools = golden_tools(row["tools_json"].as_str().expect("tools_json"));
        let thinking = golden_thinking(row);
        let expected = row["reference"].as_str().expect("reference render");
        let rendered = template
            .render(&messages, &tools, thinking)
            .expect("render");
        assert_eq!(rendered, expected, "golden render {id}");
    }
}

#[test]
fn golden_renders_have_the_framework_markers() {
    let goldens = goldens();
    let renders = golden_renders(&goldens);
    for row in renders {
        let id = row["id"].as_str().expect("id");
        let reference = row["reference"].as_str().expect("reference render");
        let index = reference
            .rfind(GENERATION_MARKER)
            .unwrap_or_else(|| panic!("golden {id} lacks the generation marker"));
        let tail = &reference[index + GENERATION_MARKER.len()..];
        assert!(
            tail == THINK_PREAMBLE || tail == EMPTY_THINK_BLOCK,
            "golden {id} has the unexpected generation tail {tail:?}"
        );
    }
    let system = renders
        .iter()
        .find(|row| row["id"] == "system-user-default")
        .expect("system fixture");
    assert!(
        system["reference"]
            .as_str()
            .expect("reference")
            .starts_with(SYSTEM_MARKER),
        "system golden lacks the system marker"
    );
}

#[test]
fn system_only_messages_raise_bad_request() {
    let template = pinned_template();
    let messages = [PromptMessage {
        role: Role::System,
        content: Some("You are terse.".to_string()),
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    }];
    let error = template
        .render(&messages, &[], Thinking::Default)
        .expect_err("system-only messages must raise");
    assert!(
        matches!(&error, EngError::BadRequest(message) if message.contains("No user query")),
        "{error:?}"
    );
}

#[test]
fn template_syntax_errors_report_checkpoint() {
    let error = ChatTemplate::from_str("{% for %}")
        .err()
        .expect("syntax error must fail");
    assert!(
        matches!(&error, EngError::Checkpoint(message) if message.contains("syntax error")),
        "{error:?}"
    );
}

#[test]
fn tool_call_arguments_strings_render_as_parameter_maps() {
    let template = ChatTemplate::from_str(concat!(
        "{% for m in messages %}{% for c in m.tool_calls %}",
        "{% for k, v in c.arguments|items %}{{ k }}={{ v }},{% endfor %}",
        "{% endfor %}{% endfor %}"
    ))
    .expect("parse template");
    let messages = [PromptMessage {
        role: Role::Assistant,
        content: None,
        name: None,
        tool_call_id: None,
        tool_calls: vec![ToolCallSpec {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: "{\"city\": \"Paris\"}".to_string(),
        }],
    }];
    let rendered = template
        .render(&messages, &[], Thinking::Default)
        .expect("render");
    assert_eq!(rendered, "city=Paris,");
}

#[test]
fn tool_call_argument_iteration_preserves_insertion_order() {
    let template = ChatTemplate::from_str(concat!(
        "{% for m in messages %}{% for c in m.tool_calls %}",
        "{% for k, v in c.arguments|items %}{{ k }}={{ v }},{% endfor %}",
        "{% endfor %}{% endfor %}"
    ))
    .expect("parse template");
    let messages = [PromptMessage {
        role: Role::Assistant,
        content: None,
        name: None,
        tool_call_id: None,
        tool_calls: vec![ToolCallSpec {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: "{\"unit\": \"celsius\", \"city\": \"Paris\"}".to_string(),
        }],
    }];
    let rendered = template
        .render(&messages, &[], Thinking::Default)
        .expect("render");
    assert_eq!(rendered, "unit=celsius,city=Paris,");
}

#[test]
fn unknown_method_serves_the_five_string_methods() {
    let template = ChatTemplate::from_str(concat!(
        "{% set s = 'a>b>c' %}",
        "{{ s.split('>')[1] }}|",
        "{{ 'xxhelloxx'.lstrip('x') }}|",
        "{{ 'xxhelloxx'.rstrip('x') }}|",
        "{{ 'hello'.startswith('he') }}|",
        "{{ 'hello'.endswith('lo') }}"
    ))
    .expect("parse template");
    let messages = [PromptMessage {
        role: Role::User,
        content: Some("hi".to_string()),
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    }];
    let rendered = template
        .render(&messages, &[], Thinking::Default)
        .expect("render");
    assert_eq!(rendered, "b|helloxx|xxhello|True|True");
}

#[test]
fn unknown_method_rejects_bad_callers_and_arguments() {
    let messages = [PromptMessage {
        role: Role::User,
        content: Some("hi".to_string()),
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    }];
    let non_string = ChatTemplate::from_str("{{ 5.startswith('a') }}").expect("parse template");
    let error = non_string
        .render(&messages, &[], Thinking::Default)
        .expect_err("non-string receiver must fail");
    assert!(
        matches!(&error, EngError::Checkpoint(message) if message.contains("unknown template method startswith")),
        "{error:?}"
    );
    let unknown = ChatTemplate::from_str("{{ 'a'.upper('x') }}").expect("parse template");
    let error = unknown
        .render(&messages, &[], Thinking::Default)
        .expect_err("unknown method must fail");
    assert!(
        matches!(&error, EngError::Checkpoint(message) if message.contains("unknown template method upper")),
        "{error:?}"
    );
}

#[test]
fn token_id_reports_known_tokens_and_none_for_unknown() {
    let tokenizer =
        Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).expect("parse fixture tokenizer");
    assert_eq!(tokenizer.token_id("hello"), Some(8));
    assert_eq!(tokenizer.token_id("no_such_token"), None);
}

#[test]
fn tool_call_arguments_that_are_not_valid_json_fail_bad_request() {
    let template = pinned_template();
    let messages = [PromptMessage {
        role: Role::Assistant,
        content: None,
        name: None,
        tool_call_id: None,
        tool_calls: vec![ToolCallSpec {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: "not json".to_string(),
        }],
    }];
    let error = template
        .render(&messages, &[], Thinking::Default)
        .expect_err("invalid JSON arguments must fail");
    assert!(
        matches!(&error, EngError::BadRequest(message) if message.starts_with("tool call arguments failed:")),
        "{error:?}"
    );
}

const LARGE_VOCAB: usize = 60_000;

fn large_vocab_tokenizer() -> Tokenizer {
    let mut vocab = String::new();
    for index in 0..LARGE_VOCAB {
        if index > 0 {
            vocab.push(',');
        }
        vocab.push_str(&format!("\"tok{index}\":{index}"));
    }
    let json = format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}},"model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{{{vocab}}},"merges":[]}}}}"#
    );
    Tokenizer::from_bytes(json.as_bytes()).expect("parse large fixture tokenizer")
}

#[test]
fn incremental_decoders_share_one_tokenizer() {
    let tokenizer = fixture_tokenizer();
    let first = IncrementalDecoder::new(&tokenizer);
    let second = IncrementalDecoder::new(&tokenizer);
    let shared = tokenizer.raw() as *const tokenizers::Tokenizer;
    assert_eq!(first.tokenizer_addr(), shared);
    assert_eq!(second.tokenizer_addr(), shared);
}

#[test]
fn decoder_push_cost_does_not_scale_with_the_vocabulary() {
    let tokenizer = large_vocab_tokenizer();
    let mut decoder = IncrementalDecoder::new(&tokenizer);
    let started = std::time::Instant::now();
    for index in 0..100 {
        decoder.push(index).expect("push");
    }
    let elapsed = started.elapsed();
    assert!(
        elapsed < std::time::Duration::from_millis(500),
        "100 pushes took {elapsed:?}; the per-push vocabulary size query is too slow"
    );
}

const REWRITING_TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"Sequence","decoders":[{"type":"Fuse"},{"type":"Replace","pattern":{"String":"ab"},"content":"X"}]},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2},"merges":[]}}"#;

#[test]
fn incremental_decoder_finish_reports_a_prefix_mismatch() {
    let tokenizer =
        Tokenizer::from_bytes(REWRITING_TOKENIZER_JSON.as_bytes()).expect("parse fixture");
    let mut decoder = IncrementalDecoder::new(&tokenizer);
    assert_eq!(decoder.push(0).expect("first token"), "a");
    assert_eq!(decoder.push(1).expect("second token"), "");
    let error = decoder
        .finish()
        .expect_err("the text no longer extends the sent text");
    assert!(
        matches!(&error, EngError::Checkpoint(message) if message.contains("prefix")),
        "error: {error:?}"
    );
}

#[test]
fn incremental_decoder_finish_drains_the_pending_tail_once() {
    let mut decoder = IncrementalDecoder::new(&split_tokenizer());
    assert_eq!(decoder.push(0).expect("first byte"), "");
    assert_eq!(decoder.finish().expect("finish"), "\u{fffd}");
    assert_eq!(decoder.finish().expect("second finish"), "");
}

#[test]
fn load_reports_every_eos_id() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pin = write_fixture(dir.path());
    let bundle = TokenizerBundle::load(dir.path(), &pin).expect("load");
    assert_eq!(bundle.eos_token_ids(), [EOS_ID, SECOND_EOS_ID]);
    assert_eq!(bundle.eos_token_id(), EOS_ID);
}

#[test]
fn generation_config_accepts_one_eos_integer() {
    assert_eq!(
        eos_ids_for(r#"{"eos_token_id":11}"#),
        [EOS_ID, SECOND_EOS_ID]
    );
}

#[test]
fn eos_ids_keep_the_tokenizer_eos_first_without_duplicates() {
    assert_eq!(
        eos_ids_for(r#"{"eos_token_id":[11,15,11]}"#),
        [EOS_ID, SECOND_EOS_ID]
    );
}

#[test]
fn generation_config_without_eos_keeps_the_tokenizer_eos() {
    assert_eq!(eos_ids_for(r#"{"temperature":1.0}"#), [EOS_ID]);
}

#[test]
fn eos_ids_outside_the_vocab_or_of_the_wrong_type_fail_checkpoint() {
    for text in [
        r#"{"eos_token_id":[15,16]}"#,
        r#"{"eos_token_id":["x"]}"#,
        r#"{"eos_token_id":1.5}"#,
        r#"{"eos_token_id":-1}"#,
        r#"{"eos_token_id":null}"#,
        r#"{"eos_token_id":[[11]]}"#,
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut pin = write_fixture(dir.path());
        rewrite_generation_config(dir.path(), &mut pin, text);
        let message = checkpoint_message(TokenizerBundle::load(dir.path(), &pin));
        assert!(message.contains("eos_token_id"), "{text}: {message}");
    }
}

#[test]
fn generation_config_is_pinned_like_the_other_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut pin = write_fixture(dir.path());
    pin.generation_config_json[0] ^= 1;
    let message = checkpoint_message(TokenizerBundle::load(dir.path(), &pin));
    assert!(message.contains("generation_config.json"), "{message}");
    let dir = tempfile::tempdir().expect("tempdir");
    let pin = write_fixture(dir.path());
    fs::remove_file(dir.path().join("generation_config.json")).expect("remove");
    let message = checkpoint_message(TokenizerBundle::load(dir.path(), &pin));
    assert!(message.contains("generation_config.json"), "{message}");
}

#[test]
fn duplicate_config_members_fail_checkpoint() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut pin = write_fixture(dir.path());
    rewrite_tokenizer_config(
        dir.path(),
        &mut pin,
        r#"{"eos_token":"<|end|>","eos_token":"hello"}"#,
    );
    let message = checkpoint_message(TokenizerBundle::load(dir.path(), &pin));
    assert!(message.contains("duplicate"), "{message}");
    let dir = tempfile::tempdir().expect("tempdir");
    let mut pin = write_fixture(dir.path());
    rewrite_generation_config(
        dir.path(),
        &mut pin,
        r#"{"eos_token_id":15,"eos_token_id":11}"#,
    );
    let message = checkpoint_message(TokenizerBundle::load(dir.path(), &pin));
    assert!(message.contains("duplicate"), "{message}");
}

#[test]
fn unreadable_pinned_file_is_not_reported_as_missing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pin = write_fixture(dir.path());
    fs::remove_file(dir.path().join("chat_template.jinja")).expect("remove");
    fs::create_dir(dir.path().join("chat_template.jinja")).expect("directory in its place");
    let message = checkpoint_message(TokenizerBundle::load(dir.path(), &pin));
    assert!(message.contains("chat_template.jinja"), "{message}");
    assert!(!message.contains("missing"), "{message}");
    assert!(message.contains("could not be read"), "{message}");
}

// Expected values come from `render_jinja_template` of transformers 5.17.0 with
// jinja2 3.1.6 on Python 3.12.14 (the pinned reference stack on the Mac). The
// probe script renders `{{ (EXPR) | tojson }}`. `None` marks a Python exception.
const STRING_METHOD_CASES: &[(&str, Option<&str>)] = &[
    (
        "'a  b\\tc\\n d '.split()",
        Some("[\"a\", \"b\", \"c\", \"d\"]"),
    ),
    ("'a,b,c'.split(',')", Some("[\"a\", \"b\", \"c\"]")),
    ("'a,b,c,d'.split(',', 1)", Some("[\"a\", \"b,c,d\"]")),
    (
        "'a,b,c,d'.split(',', maxsplit=1)",
        Some("[\"a\", \"b,c,d\"]"),
    ),
    (
        "'a,b,c,d'.split(',', -1)",
        Some("[\"a\", \"b\", \"c\", \"d\"]"),
    ),
    ("'a,b,c,d'.split(',', 0)", Some("[\"a,b,c,d\"]")),
    ("'abc'.split('')", None),
    ("' a b  c '.split(None, 1)", Some("[\"a\", \"b  c \"]")),
    ("' a b  c '.split(sep=None)", Some("[\"a\", \"b\", \"c\"]")),
    ("''.split(',')", Some("[\"\"]")),
    ("''.split()", Some("[]")),
    ("'  \\t\\nxx  '.lstrip()", Some("\"xx  \"")),
    ("'  xx \\t\\n '.rstrip()", Some("\"  xx\"")),
    ("'xyxhixyx'.lstrip('xy')", Some("\"hixyx\"")),
    ("'xyxhixyx'.rstrip('xy')", Some("\"xyxhi\"")),
    ("'  hi '.lstrip(None)", Some("\"hi \"")),
    ("'  hi '.lstrip('')", Some("\"  hi \"")),
    ("'xx'.lstrip('x','y')", None),
    ("'xx'.rstrip('x','y')", None),
    ("'hello'.startswith(('x','he'))", Some("true")),
    ("'hello'.startswith()", None),
    ("'hello'.startswith('l', 2)", Some("true")),
    ("'hello'.startswith('l', 2, 4)", Some("true")),
    ("'hello'.endswith('l', 0, 4)", Some("true")),
    ("'hello'.endswith(('x','lo'))", Some("true")),
    ("'hello'.startswith('')", Some("true")),
    ("'hello'.startswith(5)", None),
    ("'hello'.split(5)", None),
    ("'hello'.startswith('lo', -2)", Some("true")),
    ("'hello'.endswith('he', 0, -3)", Some("true")),
    ("'hello'.startswith('', 6)", Some("false")),
    ("'hello'.startswith('', 5)", Some("true")),
    ("'hello'.endswith('', 6)", Some("false")),
    ("'hello'.endswith('o', 9)", Some("false")),
    ("'hello'.startswith('', 3, 1)", Some("false")),
    ("'héllo'.startswith('llo', 2)", Some("true")),
    ("'héllo'.endswith('él', 0, 3)", Some("true")),
    ("'hello'.startswith(())", Some("false")),
    ("'hello'.startswith(('he', 5))", Some("true")),
    ("'hello'.startswith((5, 'he'))", None),
    ("'hello'.startswith('he', start=1)", None),
    ("'hello'.startswith('he', None)", Some("true")),
    ("'hello'.startswith('he', 'x')", None),
    ("'a,b'.split(sep=',')", Some("[\"a\", \"b\"]")),
    (
        "'a,b,c'.split(maxsplit=1, sep=',')",
        Some("[\"a\", \"b,c\"]"),
    ),
    ("'a,b'.split(',', sep=',')", None),
    ("'a,b'.split(',', 'x')", None),
    ("'a,b'.split(limit=1)", None),
    ("'a,b'.split(',', 1, 2)", None),
    ("'a::b::c:d'.split('::')", Some("[\"a\", \"b\", \"c:d\"]")),
    ("'  foo   '.split(maxsplit=0)", Some("[\"foo   \"]")),
    ("'   '.split(None, 0)", Some("[]")),
    (
        "'a\u{3000}b\u{a0}c\u{2028}d\u{1c}e'.split()",
        Some("[\"a\", \"b\", \"c\", \"d\", \"e\"]"),
    ),
    ("'  x '.lstrip(None)", Some("\"x \"")),
    ("'  x '.lstrip(5)", None),
    ("'x  '.rstrip(chars='x')", None),
    ("'Arthur: three!'.lstrip('Arthur: ')", Some("\"ee!\"")),
    ("'Monty Python'.rstrip(' Python')", Some("\"M\"")),
    ("'éèx'.lstrip('èé')", Some("\"x\"")),
    ("'\\x1csys\\x1c' | trim", Some("\"sys\"")),
    (
        "' \\t\\n\\x1d\\x1e\\x1fx\\u3000\\u00a0 ' | trim",
        Some("\"x\""),
    ),
    ("'  hi  ' | trim", Some("\"hi\"")),
    ("'xxhixx' | trim('x')", Some("\"hi\"")),
    ("'xyhiyx' | trim('xy')", Some("\"hi\"")),
    ("'  hi ' | trim(None)", Some("\"hi\"")),
    ("'  hi ' | trim('')", Some("\"  hi \"")),
    ("'  hi ' | trim(5)", None),
    ("5 | trim", Some("\"5\"")),
    ("none | trim", Some("\"None\"")),
    ("'  hi ' | trim('x', 'y')", None),
    ("'a,b'.split(',', None)", None),
    ("'a,b'.split(',', maxsplit=None)", None),
    ("'a b c'.split(None, 1)", Some("[\"a\", \"b c\"]")),
];

#[test]
fn string_methods_match_the_python_reference() {
    for (expr, expected) in STRING_METHOD_CASES {
        let template =
            ChatTemplate::from_str(&format!("{{{{ ({expr}) | tojson }}}}")).expect("parse");
        let result = template.render(&[user_message("hi")], &[], Thinking::Default);
        match (expected, result) {
            (Some(want), Ok(got)) => assert_eq!(&got, want, "{expr}"),
            (None, Err(EngError::Checkpoint(_))) => {}
            (want, got) => panic!("{expr}: want {want:?}, got {got:?}"),
        }
    }
}

// Expected values come from the same reference stack. The probe renders each
// template through `render_jinja_template`, which sets trim_blocks and
// lstrip_blocks. The cases that use `{% raw %}` or loop controls are not here.
const WHITESPACE_CASES: &[(&str, &str)] = &[
    ("a\n  {% if true %}\nb\n  {% endif %}\nc\n", "a\nb\nc"), // trim-lstrip-if
    (
        "{% for i in [1,2] %}\n  {{ i }}\n{% endfor %}\nend",
        "  1\n  2\nend",
    ), // for-indent
    ("x {% if true %} y {% endif %} z\n", "x  y  z"),         // inline-block
    ("{{ 'v' }}\nnext", "v\nnext"),                           // var-tag-newline
    ("a\n  {# c #}\nb\n{# d #}\nc", "a\nb\nc"),               // comment
    ("  {% set a = 1 %}\n  {{ a }}\n", "  1"),                // set-line
    ("a\n", "a"),                                             // trailing-one
    ("a\n\n", "a\n"),                                         // trailing-two
    ("a", "a"),                                               // trailing-none
    ("a\n{%- if true -%}\n  b\n{%- endif -%}\nc", "abc"),     // dash-both
    ("a\n\t\t{% if true %}\nb\n\t{% endif %}\nc", "a\nb\nc"), // tab-indent
    ("a\r\n  {% if true %}\r\nb\r\n  {% endif %}\r\nc", "a\nb\nc"), // crlf
    ("  {% if true %}t{% endif %}\n", "t"),                   // space-then-text
    (
        "{% macro m(x) %}\n  [{{ x }}]\n{% endmacro %}\n{{ m(1) }}\n{{ m(2) }}\n",
        "  [1]\n\n  [2]\n",
    ), // macro
];

#[test]
fn template_whitespace_follows_the_reference_block_settings() {
    for (source, expected) in WHITESPACE_CASES {
        let template = ChatTemplate::from_str(source).expect("parse");
        let rendered = template
            .render(&[user_message("hi")], &[], Thinking::Default)
            .expect("render");
        assert_eq!(&rendered, expected, "template {source:?}");
    }
}

// The 29 characters for which Python 3.13 (Unicode 15.1) `str.isspace()` is true.
const PYTHON_WHITESPACE: [char; 29] = [
    '\u{9}', '\u{a}', '\u{b}', '\u{c}', '\u{d}', '\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', '\u{20}',
    '\u{85}', '\u{a0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}',
    '\u{2005}', '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}',
    '\u{202f}', '\u{205f}', '\u{3000}',
];

#[test]
fn strip_and_split_use_the_python_whitespace_set() {
    let template = ChatTemplate::from_str(concat!(
        "{{ messages[0].content.lstrip() }}|{{ messages[0].content.rstrip() }}|",
        "{{ messages[0].content.split() | length }}"
    ))
    .expect("parse");
    for space in PYTHON_WHITESPACE {
        let rendered = template
            .render(
                &[user_message(&format!("{space}x{space}"))],
                &[],
                Thinking::Default,
            )
            .expect("render");
        assert_eq!(
            rendered,
            format!("x{space}|{space}x|1"),
            "U+{:04X}",
            space as u32
        );
    }
    for other in ['\u{200b}', '\u{180e}', '\u{feff}', 'x'] {
        let text = format!("{other}x{other}");
        let rendered = template
            .render(&[user_message(&text)], &[], Thinking::Default)
            .expect("render");
        assert_eq!(
            rendered,
            format!("{text}|{text}|1"),
            "U+{:04X}",
            other as u32
        );
    }
}

#[test]
fn template_raises_are_client_errors_and_other_failures_are_template_defects() {
    let messages = [user_message("hi")];
    let raised = ChatTemplate::from_str(
        "{% macro check(x) %}{{ raise_exception('bad ' ~ x) }}{% endmacro %}{{ check('input') }}",
    )
    .expect("parse");
    let error = raised
        .render(&messages, &[], Thinking::Default)
        .expect_err("raise");
    assert!(
        matches!(&error, EngError::BadRequest(message) if message.contains("bad input")),
        "{error:?}"
    );
    for source in [
        "{{ 1 + 'a' }}",
        "{{ 'a'.startswith() }}",
        "{{ 'a'.nosuchmethod() }}",
        "{{ 5.startswith('a') }}",
        "{{ 1 | nosuchfilter }}",
        "{{ 'a'.split('') }}",
    ] {
        let template = ChatTemplate::from_str(source).expect("parse");
        let error = template
            .render(&messages, &[], Thinking::Default)
            .expect_err(source);
        assert!(
            matches!(error, EngError::Checkpoint(_)),
            "{source}: {error:?}"
        );
    }
}

#[test]
fn per_job_types_move_between_threads() {
    fn assert_send<T: Send>() {}
    assert_send::<IncrementalDecoder>();
    assert_send::<OutputAssembler>();
    assert_send::<StopMatcher>();
}

// Reference: `render_jinja_template` on the Mac (transformers 5.17.0) with the
// pinned template, a system message "\x1csys\x1c", and a user message "hi".
// The template applies `|trim`, which strips with `str.strip`, so U+001C goes.
#[test]
fn trim_filter_strips_the_python_whitespace_set() {
    let messages = [
        PromptMessage {
            content: Some("\u{1c}sys\u{1c}".to_string()),
            role: Role::System,
            ..user_message("")
        },
        user_message("hi"),
    ];
    let rendered = pinned_template()
        .render(&messages, &[], Thinking::Default)
        .expect("render");
    assert_eq!(
        rendered,
        "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
}

#[test]
fn incremental_decoder_second_finish_is_empty_after_an_error() {
    let tokenizer =
        Tokenizer::from_bytes(REWRITING_TOKENIZER_JSON.as_bytes()).expect("parse fixture");
    let mut decoder = IncrementalDecoder::new(&tokenizer);
    decoder.push(0).expect("first token");
    decoder.push(1).expect("second token");
    decoder.finish().expect_err("prefix mismatch");
    assert_eq!(decoder.finish().expect("second finish"), "");
}

const TOJSON_GOLDENS_JSON: &str = include_str!("testdata/tojson-goldens.json");

fn tojson_goldens() -> serde_json::Value {
    let goldens: serde_json::Value =
        serde_json::from_str(TOJSON_GOLDENS_JSON).expect("tojson goldens parse");
    assert_eq!(goldens["schema"], "p1-tokenizer-tojson-goldens-1");
    goldens
}

// Decision 0029: the reference `tojson` is `json.dumps(x, ensure_ascii=False,
// sort_keys=False)`. The goldens come from `tokenizer.apply_chat_template` of
// transformers 5.17.0 on the Mac (`testdata/capture-tojson-goldens.py`). Their
// tools keep the OpenAI wrapper and an unsorted key order, and their text has
// non-ASCII characters, apostrophes, angle brackets, ampersands, and DEL.
#[test]
fn tool_prompts_match_the_reference_renders() {
    let goldens = tojson_goldens();
    let template = pinned_template();
    let rows = goldens["renders"].as_array().expect("renders");
    assert_eq!(rows.len(), 4);
    for row in rows {
        let id = row["id"].as_str().expect("id");
        let messages = golden_messages(row["messages_json"].as_str().expect("messages_json"));
        let tools = golden_tools(row["tools_json"].as_str().expect("tools_json"));
        let rendered = template
            .render(&messages, &tools, golden_thinking(row))
            .expect("render");
        assert_eq!(
            rendered,
            row["reference"].as_str().expect("reference"),
            "golden render {id}"
        );
    }
}

// Each row renders its template with `render_jinja_template`. A row with an
// `error` is a Python exception, which is a template defect here. A row with
// `known_divergence` records a difference that decision 0029 accepts. The
// table below pins what the engine prints for it.
const KNOWN_DIVERGENCES: &[(&str, Option<&str>)] = &[
    ("big-integer", Some("{\"n\": 1.2345678901234568e+22}")),
    ("string-bad-indent", None),
    ("list-display", Some("[1, \"a\", None, True]")),
];

#[test]
fn tojson_filter_matches_the_reference_for_every_argument_form() {
    let goldens = tojson_goldens();
    let rows = goldens["filters"].as_array().expect("filters");
    assert_eq!(rows.len(), 40);
    let mut divergent = 0;
    for row in rows {
        let id = row["id"].as_str().expect("id");
        let value: serde_json::Value =
            serde_json::from_str(row["x_json"].as_str().expect("x_json")).expect("x_json");
        let template =
            ChatTemplate::from_str(row["template"].as_str().expect("template")).expect("parse");
        let result = template.render(&[user_message("hi")], &[ToolSpec(value)], Thinking::Default);
        if row.get("known_divergence").is_some() {
            divergent += 1;
            let (_, engine) = KNOWN_DIVERGENCES
                .iter()
                .find(|(name, _)| *name == id)
                .expect("divergent row is pinned");
            match (engine, result) {
                (Some(want), Ok(got)) => assert_eq!(&got, want, "divergent row {id}"),
                (None, Err(EngError::Checkpoint(_))) => {}
                (want, got) => panic!("divergent row {id}: want {want:?}, got {got:?}"),
            }
            continue;
        }
        match (row["reference"].as_str(), result) {
            (Some(want), Ok(got)) => assert_eq!(got, want, "filter row {id}"),
            (None, Err(EngError::Checkpoint(_))) => {}
            (want, got) => panic!("filter row {id}: want {want:?}, got {got:?}"),
        }
    }
    assert_eq!(divergent, KNOWN_DIVERGENCES.len());
}

fn render_error_message(source: &str) -> String {
    let template = ChatTemplate::from_str(source).expect("parse template");
    match template.render(&[user_message("hi")], &[], Thinking::Default) {
        Err(EngError::Checkpoint(message)) => message,
        other => panic!("expected Checkpoint for {source}, got {other:?}"),
    }
}

#[test]
fn tokenizer_from_bytes_rejects_malformed_json() {
    match Tokenizer::from_bytes(b"not json") {
        Err(EngError::Checkpoint(message)) => {
            assert!(
                message.starts_with("tokenizer.json parse failed: "),
                "message: {message}"
            );
        }
        other => panic!("expected Checkpoint, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn tojson_rejects_an_argument_given_by_position_and_name() {
    let message = render_error_message("{{ 1 | tojson(true, ensure_ascii=true) }}");
    assert!(
        message.contains("TypeError: tojson() got multiple values for argument 'ensure_ascii'"),
        "message: {message}"
    );
}

#[test]
fn tojson_rejects_an_indent_above_the_limit() {
    let message = render_error_message("{{ 1 | tojson(indent=1000000) }}");
    assert!(
        message.contains("ValueError: tojson() indent above "),
        "message: {message}"
    );
    assert!(message.contains(" is not supported"), "message: {message}");
}

#[test]
fn tojson_rejects_non_string_separators() {
    let message = render_error_message("{{ 1 | tojson(separators=(1, 2)) }}");
    assert!(
        message.contains("TypeError: tojson() separators must hold strings"),
        "message: {message}"
    );
}

#[test]
fn string_affix_rejects_wrong_argument_counts() {
    let message = render_error_message("{{ 'a'.startswith() }}");
    assert!(
        message.contains("TypeError: startswith() takes at least 1 argument (0 given)"),
        "message: {message}"
    );
    let message = render_error_message("{{ 'a'.endswith('a', 0, 1, 2) }}");
    assert!(
        message.contains("TypeError: endswith() takes at most 3 arguments (4 given)"),
        "message: {message}"
    );
}

/// Loads the toy tokenizer bundle with `template` as its chat template.
fn bundle_with_template(template: &str) -> TokenizerBundle {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut pin = write_fixture(dir.path());
    fs::write(dir.path().join("chat_template.jinja"), template).expect("rewrite template");
    pin.chat_template_jinja = digest(template.as_bytes());
    TokenizerBundle::load(dir.path(), &pin).expect("load")
}

#[test]
fn the_pinned_template_without_a_generation_prompt_ends_at_the_message_boundary() {
    let template = pinned_template();
    let messages = [user_message("hi")];
    let head = template
        .render_without_generation_prompt(&messages, &[], Thinking::Off)
        .expect("render head");
    assert_eq!(head, "\u{3c}|im_start|\u{3e}user\nhi\u{3c}|im_end|\u{3e}\n");
    let off = template
        .render(&messages, &[], Thinking::Off)
        .expect("render off");
    let on = template
        .render(&messages, &[], Thinking::On)
        .expect("render on");
    assert_eq!(off, format!("{head}{GENERATION_MARKER}{EMPTY_THINK_BLOCK}"));
    assert_eq!(on, format!("{head}{GENERATION_MARKER}{THINK_PREAMBLE}"));
    let without_on = template
        .render_without_generation_prompt(&messages, &[], Thinking::On)
        .expect("render head on");
    assert_eq!(without_on, head, "the head does not depend on thinking");
}

#[test]
fn the_pinned_template_keeps_a_tool_result_inside_the_head() {
    let template = pinned_template();
    let tool_result = PromptMessage {
        role: Role::Tool,
        content: Some("5".to_string()),
        name: None,
        tool_call_id: Some("call-1".to_string()),
        tool_calls: vec![],
    };
    let messages = [user_message("go"), tool_result];
    let head = template
        .render_without_generation_prompt(&messages, &[], Thinking::Off)
        .expect("render head");
    assert!(
        head.ends_with("\n<tool_response>\n5\n</tool_response>\u{3c}|im_end|\u{3e}\n"),
        "head: {head:?}"
    );
    let full = template
        .render(&messages, &[], Thinking::Off)
        .expect("render");
    assert_eq!(
        full,
        format!("{head}{GENERATION_MARKER}{EMPTY_THINK_BLOCK}")
    );
}

#[test]
fn the_boundary_is_the_id_count_of_the_text_before_the_generation_tail() {
    let bundle = bundle_with_template(
        "{{ messages[0].content }}{% if add_generation_prompt %} world{% endif %}",
    );
    let messages = [user_message("hello")];
    let (ids, boundary) = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("encode");
    assert_eq!(ids, vec![8, 14], "ids of hello and the tail");
    assert_eq!(boundary, Some(1), "the head holds the id of hello");
    let plain = bundle
        .encode_prompt(&messages, &[], Thinking::Default)
        .expect("encode");
    assert_eq!(ids, plain, "same ids as encode_prompt");
}

#[test]
fn the_boundary_is_none_when_the_tail_merges_into_the_text_before_it() {
    // "he" encodes to [11], but "he" followed by "llo world" encodes to [8, 14]: the ids do not
    // start with [11].
    let bundle = bundle_with_template(
        "{{ messages[0].content }}{% if add_generation_prompt %}llo world{% endif %}",
    );
    let (ids, boundary) = bundle
        .encode_prompt_with_boundary(&[user_message("he")], &[], Thinking::Default)
        .expect("encode");
    assert_eq!(ids, vec![8, 14], "ids of hello world");
    assert_eq!(boundary, None);
}

#[test]
fn the_boundary_is_none_when_the_generation_prompt_comes_first() {
    let bundle =
        bundle_with_template("{% if add_generation_prompt %}a{% endif %}{{ messages[0].content }}");
    let (ids, boundary) = bundle
        .encode_prompt_with_boundary(&[user_message("hello")], &[], Thinking::Default)
        .expect("encode");
    assert_eq!(ids, vec![9, 8], "ids of a and hello");
    assert_eq!(boundary, None);
}

#[test]
fn the_boundary_is_none_when_nothing_precedes_the_tail() {
    let bundle = bundle_with_template("{% if add_generation_prompt %}hello{% endif %}");
    let (ids, boundary) = bundle
        .encode_prompt_with_boundary(&[user_message("x")], &[], Thinking::Default)
        .expect("encode");
    assert_eq!(ids, vec![8], "ids of the tail");
    assert_eq!(boundary, None);
}

#[test]
fn the_boundary_is_none_when_the_tail_adds_no_token() {
    let bundle =
        bundle_with_template("{{ messages[0].content }}{% if add_generation_prompt %}{% endif %}");
    let (ids, boundary) = bundle
        .encode_prompt_with_boundary(&[user_message("hello")], &[], Thinking::Default)
        .expect("encode");
    assert_eq!(ids, vec![8], "ids of hello");
    assert_eq!(boundary, None);
}

#[test]
fn the_boundary_follows_the_tools_that_the_head_renders() {
    // The template prints "a" only when a tool exists. Hand-derived ids of the toy vocabulary:
    // "ahello" is [9, 8] (a, then h+e+l+l+o merged to hello), "hello" is [8], " world" is [14].
    let bundle = bundle_with_template(concat!(
        "{% if tools %}a{% endif %}{{ messages[0].content }}",
        "{% if add_generation_prompt %} world{% endif %}",
    ));
    let messages = [user_message("hello")];
    let (ids, boundary) = bundle
        .encode_prompt_with_boundary(&messages, &[example_tool()], Thinking::Default)
        .expect("encode with a tool");
    assert_eq!(ids, vec![9, 8, 14], "ids of a, hello, and the tail");
    assert_eq!(
        boundary,
        Some(2),
        "the head holds the tool marker and hello"
    );
    let (ids, boundary) = bundle
        .encode_prompt_with_boundary(&messages, &[], Thinking::Default)
        .expect("encode without a tool");
    assert_eq!(ids, vec![8, 14], "ids of hello and the tail");
    assert_eq!(boundary, Some(1), "the head holds hello alone");
}

#[test]
fn the_boundary_follows_the_thinking_mode_that_the_head_renders() {
    // The head prints "a" for On, "b" for Off, and nothing for Default. "bhello" is [10, 8].
    let bundle = bundle_with_template(concat!(
        "{% if enable_thinking is defined %}{% if enable_thinking %}a{% else %}b{% endif %}",
        "{% endif %}{{ messages[0].content }}{% if add_generation_prompt %} world{% endif %}",
    ));
    let messages = [user_message("hello")];
    let cases = [
        (Thinking::Default, vec![8, 14], Some(1)),
        (Thinking::On, vec![9, 8, 14], Some(2)),
        (Thinking::Off, vec![10, 8, 14], Some(2)),
    ];
    for (thinking, want_ids, want_boundary) in cases {
        let (ids, boundary) = bundle
            .encode_prompt_with_boundary(&messages, &[], thinking)
            .expect("encode");
        assert_eq!(ids, want_ids, "ids for {thinking:?}");
        assert_eq!(boundary, want_boundary, "boundary for {thinking:?}");
    }
}

#[test]
fn the_pinned_template_renders_the_tools_block_inside_the_head() {
    let template = pinned_template();
    let tool = ToolSpec(serde_json::json!({"name": "f"}));
    let messages = [user_message("hi")];
    let head = template
        .render_without_generation_prompt(&messages, std::slice::from_ref(&tool), Thinking::Off)
        .expect("render head");
    assert!(
        head.starts_with(concat!(
            "\u{3c}|im_start|\u{3e}system\n# Tools\n\nYou have access to the following functions:",
            "\n\n\u{3c}tools\u{3e}\n{\"name\": \"f\"}\n\u{3c}/tools\u{3e}\n\nIf you choose"
        )),
        "head: {head:?}"
    );
    assert!(
        head.ends_with(concat!(
            "\u{3c}/IMPORTANT\u{3e}\u{3c}|im_end|\u{3e}\n",
            "\u{3c}|im_start|\u{3e}user\nhi\u{3c}|im_end|\u{3e}\n"
        )),
        "head: {head:?}"
    );
    let full = template
        .render(&messages, std::slice::from_ref(&tool), Thinking::Off)
        .expect("render");
    assert_eq!(
        full,
        format!("{head}{GENERATION_MARKER}{EMPTY_THINK_BLOCK}")
    );
    let without_tool = template
        .render_without_generation_prompt(&messages, &[], Thinking::Off)
        .expect("render head without a tool");
    assert!(
        !head.eq(&without_tool) && head.ends_with(&without_tool),
        "the tools block comes first and the messages follow it"
    );
}

#[test]
fn the_pinned_template_keeps_a_trailing_assistant_message_inside_the_head() {
    let template = pinned_template();
    let assistant = PromptMessage {
        role: Role::Assistant,
        content: Some("ok".to_string()),
        name: None,
        tool_call_id: None,
        tool_calls: vec![],
    };
    let messages = [user_message("hi"), assistant];
    let head = template
        .render_without_generation_prompt(&messages, &[], Thinking::Off)
        .expect("render head");
    assert_eq!(
        head,
        concat!(
            "\u{3c}|im_start|\u{3e}user\nhi\u{3c}|im_end|\u{3e}\n",
            "\u{3c}|im_start|\u{3e}assistant\n\u{3c}think\u{3e}\n\n\u{3c}/think\u{3e}\n\nok",
            "\u{3c}|im_end|\u{3e}\n"
        )
    );
    let full = template
        .render(&messages, &[], Thinking::Off)
        .expect("render");
    assert_eq!(
        full,
        format!("{head}{GENERATION_MARKER}{EMPTY_THINK_BLOCK}")
    );
}
