//! Tests of cancellation through the whole stack: client, router, handler, pump, and engine thread.
//!
//! Rules: the server design ("Streaming format"). A client disconnect cancels the job. A job
//! whose bounded output channel is full with no reader is cancelled. Neither case sends `[DONE]`
//! or an error event.

use super::router::router;
use super::state::AppState;
use super::testkit::{self, Probes};
use crate::sched::job::{Job, JobEvent};
use crate::sched::worker::EngineState;
use axum::Router;
use axum::http::{StatusCode, header};
use futures::StreamExt;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

fn app(probes: &Probes) -> Router {
    testkit::wait_for_state(&probes.engine.handle(), EngineState::Ready);
    let entry = testkit::entry("e1", "model-a", &probes.engine, 100_000);
    router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])))
}

fn request(extra: Value) -> Value {
    let mut body = json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}]});
    for (key, value) in extra.as_object().expect("object").iter() {
        body[key] = value.clone();
    }
    body
}

/// Decodes after which a job counts as not cancelled. A job with a 2 ms decode delay reaches this
/// bound in about one second, inside the five second bound of `wait_until`.
const RUN_ON_LIMIT: usize = 500;

/// Waits until a job that has decoded at least once has ended, or until it has run on past
/// `RUN_ON_LIMIT` decodes. Then it checks the engine state and the decode count, so a job that is
/// not cancelled fails an assertion instead of the wait timeout.
fn wait_for_the_job_to_stop(probes: &Probes) {
    let handle = probes.engine.handle();
    testkit::wait_until("the job to stop or to run on", || {
        let decodes = probes.decodes.load(Ordering::SeqCst);
        (decodes > 0 && handle.state() == EngineState::Ready) || decodes >= RUN_ON_LIMIT
    });
    let decodes = probes.decodes.load(Ordering::SeqCst);
    assert_eq!(
        handle.state(),
        EngineState::Ready,
        "the job still runs after the client left: {decodes} decodes"
    );
    assert!(
        decodes < RUN_ON_LIMIT,
        "the job ran on after the client left: {decodes} decodes"
    );
}

/// Sends a short request and checks that the engine answers it with its own first token only.
async fn assert_the_next_job_is_unaffected(app: &Router) {
    let reuse = request(json!({"temperature": 0, "max_tokens": 1}));
    let (status, _headers, bytes) =
        testkit::send(app, testkit::post_json("/v1/chat/completions", &reuse)).await;
    assert_eq!(status, StatusCode::OK);
    let reply = testkit::json_of(&bytes);
    assert_eq!(reply["choices"][0]["message"]["content"], "\u{1}");
}

#[tokio::test]
async fn client_drop_cancels_job_and_stops_generation() {
    let probes = testkit::probes(Duration::from_millis(2));
    let app = app(&probes);
    let body = request(json!({"stream": true, "max_tokens": 50_000}));
    let response = tower::ServiceExt::oneshot(
        app.clone(),
        testkit::post_json("/v1/chat/completions", &body),
    )
    .await
    .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let mut frames = response.into_body().into_data_stream();
    let first = frames.next().await.expect("a frame").expect("frame bytes");
    assert!(first.starts_with(b"data: "), "{first:?}");
    testkit::wait_until("the first decode", || {
        probes.decodes.load(Ordering::SeqCst) > 0
    });
    drop(frames);
    wait_for_the_job_to_stop(&probes);
    assert_the_next_job_is_unaffected(&app).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropped_json_request_stops_the_running_job() {
    let probes = testkit::probes(Duration::from_millis(2));
    let app = app(&probes);
    let body = request(json!({"max_tokens": 50_000}));
    let pending = {
        let app = app.clone();
        tokio::spawn(async move {
            testkit::send(&app, testkit::post_json("/v1/chat/completions", &body)).await
        })
    };
    testkit::wait_until("the first decode", || {
        probes.decodes.load(Ordering::SeqCst) > 0
    });
    pending.abort();
    wait_for_the_job_to_stop(&probes);
    assert_the_next_job_is_unaffected(&app).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropped_request_cancels_a_queued_job_before_it_starts() {
    let (probes, gate) = testkit::gated_probes();
    let app = {
        let entry = testkit::entry("e1", "model-a", &probes.engine, 100_000);
        router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])))
    };
    let body = request(json!({"max_tokens": 50_000}));
    let pending = tokio::spawn(async move {
        testkit::send(&app, testkit::post_json("/v1/chat/completions", &body)).await
    });
    let handle = probes.engine.handle();
    // The engine is still loading, so the job waits in the queue. Give the handler time to queue it.
    // The kit offers no observable for the submit: the queue length is private, and the probes
    // record nothing before the engine loads. A late submit makes the test pass without the queued
    // path, never fail. So a fixed wait stays.
    tokio::time::sleep(Duration::from_millis(100)).await;
    pending.abort();
    let _ = pending.await;
    drop(gate);
    testkit::wait_for_state(&handle, EngineState::Ready);
    // The queue is first in, first out. When the sentinel job has a first batch, the cancelled job
    // before it has been passed over.
    let (job, mut sentinel) = Job::new(vec![104], testkit::generation()).expect("job");
    handle.try_submit(job).expect("submit");
    let first = sentinel.events.recv().await;
    assert!(matches!(first, Some(JobEvent::Batch(_))), "{first:?}");
    sentinel.cancel();
    let prefilled = probes.prompt.lock().expect("prompt lock").clone();
    assert_eq!(
        prefilled,
        vec![104],
        "the cancelled job was prefilled: {prefilled:?}"
    );
}

#[tokio::test]
async fn a_full_output_channel_cancels_the_job_and_the_stream_closes_without_done_or_error() {
    let probes = testkit::probes(Duration::ZERO);
    let app = app(&probes);
    let body = request(json!({"stream": true, "max_tokens": 50_000}));
    let response = tower::ServiceExt::oneshot(
        app.clone(),
        testkit::post_json("/v1/chat/completions", &body),
    )
    .await
    .expect("response");
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/event-stream"
    );
    let handle = probes.engine.handle();
    testkit::wait_until("the engine to stop the unread job", || {
        probes.decodes.load(Ordering::SeqCst) > 1_000 && handle.state() == EngineState::Ready
    });
    let stopped_at = probes.decodes.load(Ordering::SeqCst);
    assert!(stopped_at < 1_100, "the job ran on: {stopped_at} decodes");
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(probes.decodes.load(Ordering::SeqCst), stopped_at);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let events = testkit::sse_events(&bytes);
    assert!(
        events.len() > 100,
        "the buffered events are lost: {}",
        events.len()
    );
    assert!(events.iter().all(|event| event.name.is_none()));
    assert!(events.iter().all(|event| event.data != "[DONE]"));
    assert!(
        events
            .iter()
            .all(|event| !event.data.contains("finish_reason"))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_that_closes_its_socket_cancels_the_job() {
    let probes = testkit::probes(Duration::from_millis(2));
    let app = app(&probes);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    let served = app.clone();
    let server = tokio::spawn(async move { axum::serve(listener, served).await });
    let body = request(json!({"stream": true, "max_tokens": 50_000})).to_string();
    let client = tokio::task::spawn_blocking(move || {
        let mut socket = std::net::TcpStream::connect(address).expect("connect");
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("timeout");
        let head = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            testkit::API_KEY,
            body.len()
        );
        socket.write_all(head.as_bytes()).expect("write head");
        socket.write_all(body.as_bytes()).expect("write body");
        let mut received = Vec::new();
        let mut chunk = [0_u8; 512];
        // The role chunk comes first. The second data event shows that the job decodes.
        while String::from_utf8_lossy(&received)
            .matches("data: {")
            .count()
            < 2
        {
            let count = socket.read(&mut chunk).expect("read");
            assert!(count > 0, "the server closed before the first event");
            received.extend_from_slice(&chunk[..count]);
        }
        String::from_utf8_lossy(&received).to_string()
    });
    let text = client.await.expect("client thread");
    assert!(text.starts_with("HTTP/1.1 200"), "{text}");
    wait_for_the_job_to_stop(&probes);
    server.abort();
    assert_the_next_job_is_unaffected(&app).await;
}
