//! Real pinned tokenizer through HTTP request conversion, decode, parsing, and wire replies.
use super::{router::router, state::AppState, testkit};
use crate::exec_fake::FakeExecutor;
use crate::sched::worker::{Engine, EngineState, spawn_engine};
use crate::tokenizer::{TokenizerBundle, flint_tests};
use axum::{Router, http::StatusCode};
use serde_json::{Value, json};
use std::sync::Arc;

fn app(engine: &Engine, bundle: Arc<TokenizerBundle>) -> Router {
    testkit::wait_for_state(&engine.handle(), EngineState::Ready);
    let mut entry = testkit::entry("e1", "flint", engine, 4096);
    entry.bundle = bundle;
    router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])))
}

fn scripted_app(text: &str, eos: u32) -> (Engine, Router) {
    let bundle = Arc::new(flint_tests::load_bundle());
    let mut script: Vec<_> = bundle
        .tokenizer()
        .encode(text)
        .unwrap()
        .into_iter()
        .map(|id| vec![id])
        .collect();
    script.push(vec![eos]);
    let eos_ids = bundle.eos_token_ids().to_vec();
    let engine = spawn_engine(512, move || {
        Ok((
            FakeExecutor::new(script, &eos_ids, 4096),
            Default::default(),
        ))
    })
    .unwrap();
    let router = app(&engine, bundle);
    (engine, router)
}

fn request() -> Value {
    json!({"model":"flint", "max_tokens":512, "messages":[{"role":"user","content":"Run it."}],
    "tools":[{"type":"function","function":{"name":"run","parameters":{
        "type":"object","properties":{"code":{"type":"string"},"count":{"type":"integer"}}
    }}}]})
}

#[tokio::test]
#[ignore = "requires FLINT_MODEL_DIR with the pinned publisher tokenizer"]
async fn flint_http_history_preserves_publisher_developer_reasoning_and_tool_ids() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../testdata/flint/tokenizer/goldens.json")).unwrap();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["name"] == "tool-history")
        .unwrap();
    let mut messages = case["messages"].clone();
    for message in messages.as_array_mut().unwrap() {
        if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
            for call in calls {
                *call = json!({"id":call["id"],"type":"function","function":{
                    "name":call["name"],"arguments":call["arguments"].to_string()
                }});
            }
        }
    }
    let (engine, seen) = testkit::probe_engine();
    let router = app(&engine, Arc::new(flint_tests::load_bundle()));
    let body = json!({"model":"flint","max_tokens":1,"messages":messages,"tools":case["tools"]});
    let (status, _, bytes) =
        testkit::send(&router, testkit::post_json("/v1/chat/completions", &body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let expected: Vec<u32> = serde_json::from_value(case["references"][1]["ids"].clone()).unwrap();
    assert_eq!(*seen.lock().unwrap(), expected);
}

// The existing wire adapter emits an id/name delta followed by argument deltas.
fn joined_call(deltas: Vec<Value>) -> Vec<Value> {
    assert!(!deltas.is_empty());
    assert!(deltas.iter().all(|call| call["index"] == 0));
    let arguments: String = deltas
        .iter()
        .filter_map(|call| call["function"]["arguments"].as_str())
        .collect();
    let mut call = deltas[0].clone();
    call["function"]["arguments"] = json!(arguments);
    vec![call]
}

#[tokio::test]
#[ignore = "requires FLINT_MODEL_DIR with the pinned publisher tokenizer"]
async fn flint_http_tools_stream_and_whole_replies_share_reasoning_and_typed_arguments() {
    let text = concat!(
        "<think>Need λ.</think>Running.",
        "<tool_call>run<arg_key>code</arg_key><arg_value>print(\"λ\")</arg_value>",
        "<arg_key>count</arg_key><arg_value>2</arg_value></tool_call>"
    );
    for eos in [150_001, 150_003, 150_005, 150_018] {
        for stream in [false, true] {
            let (_engine, router) = scripted_app(text, eos);
            let mut body = request();
            body["stream"] = json!(stream);
            // The final assistant role marker must not expose this historical literal
            // as a generation-time reasoning preamble when special tokens are skipped.
            body["messages"] = json!([
                {"role":"user","content":"Run it."},
                {"role":"assistant","content":"Previous literal <think>\n"}
            ]);
            let (status, _, bytes) =
                testkit::send(&router, testkit::post_json("/v1/chat/completions", &body)).await;
            assert_eq!(status, StatusCode::OK);
            let (reasoning, content, calls, finish) = if stream {
                let events = testkit::sse_events(&bytes);
                assert_eq!(events.last().unwrap().data, "[DONE]");
                let mut reasoning = String::new();
                let mut content = String::new();
                let mut calls = Vec::new();
                let mut finish = String::new();
                for event in &events[..events.len() - 1] {
                    assert!(event.name.is_none(), "{event:?}");
                    let row: Value = serde_json::from_str(&event.data).unwrap();
                    let delta = &row["choices"][0]["delta"];
                    reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or(""));
                    content.push_str(delta["content"].as_str().unwrap_or(""));
                    if let Some(items) = delta["tool_calls"].as_array() {
                        calls.extend(items.iter().cloned());
                    }
                    if let Some(value) = row["choices"][0]["finish_reason"].as_str() {
                        finish = value.into();
                    }
                }
                (reasoning, content, joined_call(calls), finish)
            } else {
                let reply = testkit::json_of(&bytes);
                let choice = &reply["choices"][0];
                let message = &choice["message"];
                (
                    message["reasoning_content"].as_str().unwrap().into(),
                    message["content"].as_str().unwrap().into(),
                    message["tool_calls"].as_array().unwrap().clone(),
                    choice["finish_reason"].as_str().unwrap().into(),
                )
            };
            assert_eq!(reasoning, "Need λ.");
            assert_eq!(content, "Running.");
            assert_eq!(finish, "tool_calls");
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0]["id"], "call_0000000000000001_0");
            assert_eq!(calls[0]["function"]["name"], "run");
            assert_eq!(
                serde_json::from_str::<Value>(calls[0]["function"]["arguments"].as_str().unwrap())
                    .unwrap(),
                json!({"code":"print(\"λ\")","count":2})
            );
        }
    }
}

#[tokio::test]
#[ignore = "requires FLINT_MODEL_DIR with the pinned publisher tokenizer"]
async fn flint_http_no_tools_stop_and_length_do_not_publish_spurious_calls() {
    let call =
        "<tool_call>run<arg_key>code</arg_key><arg_value>print(\"λ\")</arg_value></tool_call>";
    let text = format!("OK{call}");
    for tool_choice_none in [false, true] {
        let (_engine, router) = scripted_app(&text, 150_001);
        let mut body = request();
        if tool_choice_none {
            body["tool_choice"] = json!("none");
        } else {
            body.as_object_mut().unwrap().remove("tools");
        }
        let (status, _, bytes) =
            testkit::send(&router, testkit::post_json("/v1/chat/completions", &body)).await;
        assert_eq!(status, StatusCode::OK);
        let reply = testkit::json_of(&bytes);
        assert_eq!(reply["choices"][0]["message"]["content"], text);
        assert_eq!(reply["choices"][0]["finish_reason"], "stop");
        assert!(reply["choices"][0]["message"].get("tool_calls").is_none());
    }
    let partial = "OK<tool_call>run<arg_key>code</arg_key><arg_value>before";
    let text = format!("{partial}STOPafter</arg_value></tool_call>");
    let cap = flint_tests::bundle()
        .tokenizer()
        .encode(partial)
        .unwrap()
        .len();
    for length in [false, true] {
        let (_engine, router) = scripted_app(&text, 150_001);
        let mut body = request();
        if length {
            body["max_tokens"] = json!(cap);
        } else {
            body["stop"] = json!(["STOP"]);
        }
        let (status, _, bytes) =
            testkit::send(&router, testkit::post_json("/v1/chat/completions", &body)).await;
        assert_eq!(status, StatusCode::OK);
        let reply = testkit::json_of(&bytes);
        assert_eq!(
            reply["choices"][0]["finish_reason"],
            if length { "length" } else { "stop" }
        );
        assert_eq!(
            reply["choices"][0]["message"]["content"],
            if length { "OK" } else { partial }
        );
        assert!(reply["choices"][0]["message"].get("tool_calls").is_none());
    }
}
