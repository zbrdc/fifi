//! Shared test helpers for the server modules (test builds only).

use super::fake::load_bundle;
use super::state::EngineEntry;
use crate::EngError;
use crate::exec_fake::FakeExecutor;
use crate::family::executor::{ExecutorOps, GenerationState};
use crate::family::sampler::SamplingParams;
use crate::sched::worker::{Engine, EngineHandle, EngineState, spawn_engine};
use crate::tokenizer::{DecodeBatch, DecodeFinish, IncrementalDecoder, TokenizerBundle, ToolSpec};
use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, Request, StatusCode, header};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tower::ServiceExt;

/// The bearer key of every test server.
pub(super) const API_KEY: &str = "sk-test";

/// The end-of-sequence id of the fixture tokenizer.
pub(super) const EOS: u32 = 256;

/// Loads the fixture tokenizer in `testdata/tokenizer`.
/// Its ids 0 to 255 are the byte values. Id 256 is `<|im_end|>` and id 257 is `<|im_start|>`.
pub(super) fn bundle() -> Arc<TokenizerBundle> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server/testdata/tokenizer");
    Arc::new(load_bundle(&dir).expect("load fixture tokenizer"))
}

/// The token ids of ASCII text under the fixture tokenizer.
pub(super) fn ids(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

/// A fake executor script: one batch per byte of `text`, then a batch with the EOS id.
pub(super) fn script(text: &str) -> Vec<Vec<u32>> {
    let mut batches: Vec<Vec<u32>> = ids(text).into_iter().map(|id| vec![id]).collect();
    batches.push(vec![EOS]);
    batches
}

/// Starts an engine thread on the scripted fake executor.
pub(super) fn engine(script: Vec<Vec<u32>>) -> Engine {
    spawn_engine(512, move || {
        Ok((
            FakeExecutor::new(script, &[EOS], 100_000),
            Default::default(),
        ))
    })
    .expect("spawn engine")
}

/// Starts an engine whose load fails, so its state becomes `Error`.
pub(super) fn failing_engine() -> Engine {
    spawn_engine::<FakeExecutor, _>(512, || {
        Err(EngError::Checkpoint("fixture load failure".to_string()))
    })
    .expect("spawn engine")
}

/// Starts an engine whose load waits for the returned sender to drop, so submitted jobs queue.
pub(super) fn gated_engine(script: Vec<Vec<u32>>) -> (Engine, std::sync::mpsc::Sender<()>) {
    let (gate, wait) = std::sync::mpsc::channel::<()>();
    let engine = spawn_engine(512, move || {
        let _ = wait.recv();
        Ok((
            FakeExecutor::new(script, &[EOS], 100_000),
            Default::default(),
        ))
    })
    .expect("spawn engine");
    (engine, gate)
}

/// Builds the routing entry of an engine with the contract sampling defaults.
pub(super) fn entry(id: &str, model: &str, engine: &Engine, context_limit: u32) -> EngineEntry {
    entry_with_defaults(
        id,
        model,
        engine,
        context_limit,
        crate::family::sampler::SamplingDefaults::CONTRACT,
    )
}

/// Builds the routing entry of an engine with its own sampling defaults, the values that its
/// model directory's `generation_config.json` gives.
pub(super) fn entry_with_defaults(
    id: &str,
    model: &str,
    engine: &Engine,
    context_limit: u32,
    defaults: crate::family::sampler::SamplingDefaults,
) -> EngineEntry {
    EngineEntry {
        id: id.to_string(),
        model: model.to_string(),
        handle: engine.handle(),
        bundle: bundle(),
        context_limit,
        prefix_reuse: false,
        defaults,
    }
}

/// Waits up to five seconds for an engine to reach `wanted`.
pub(super) fn wait_for_state(handle: &EngineHandle, wanted: EngineState) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while handle.state() != wanted {
        assert!(
            Instant::now() < deadline,
            "engine stayed in {:?}, wanted {wanted:?}",
            handle.state()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// A `GET` request. `key` becomes the bearer token when it is set.
pub(super) fn get(uri: &str, key: Option<&str>) -> Request<Body> {
    request("GET", uri, key)
}

/// An empty-body request with the given method. `key` becomes the bearer token when it is set.
pub(super) fn request(method: &str, uri: &str, key: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(key) = key {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    builder.body(Body::empty()).expect("build request")
}

/// Sends one request through a clone of the router and reads the whole body.
pub(super) async fn send(
    router: &Router,
    request: Request<Body>,
) -> (StatusCode, HeaderMap, Bytes) {
    let response = router
        .clone()
        .oneshot(request)
        .await
        .expect("route the request");
    let (parts, body) = response.into_parts();
    let bytes = axum::body::to_bytes(body, usize::MAX)
        .await
        .expect("read the body");
    (parts.status, parts.headers, bytes)
}

/// Parses a JSON body.
pub(super) fn json_of(bytes: &Bytes) -> Value {
    serde_json::from_slice(bytes).expect("json body")
}

/// Starts an engine whose executor fails every `prefill` with `error`.
pub(super) fn refusing_engine(error: EngError) -> Engine {
    spawn_engine(512, move || {
        Ok((RefusingExecutor(Some(error)), Default::default()))
    })
    .expect("spawn engine")
}

/// An executor that returns one scripted error from `prefill`.
struct RefusingExecutor(Option<EngError>);

impl ExecutorOps for RefusingExecutor {
    type Session = ();

    fn history<'s>(&self, _st: &'s ()) -> &'s [u32] {
        &[]
    }

    fn reset(&mut self, _st: &mut ()) -> Result<(), EngError> {
        Ok(())
    }

    fn prefill(&mut self, _st: &mut (), _toks: &[u32]) -> Result<(), EngError> {
        Err(self
            .0
            .take()
            .unwrap_or(EngError::BadRequest("refused".to_string())))
    }

    fn decode(
        &mut self,
        _st: &mut (),
        _generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        Err(EngError::BadRequest(
            "decode after a refused prefill".to_string(),
        ))
    }

    fn boundary(&mut self, _st: &mut ()) {}
}

/// A fresh generation state over the fixture tokenizer, with a greedy sampler.
pub(super) fn generation() -> GenerationState {
    GenerationState::new(IncrementalDecoder::new(bundle().tokenizer()))
}

/// A tool named `add` with an integer parameter `a` and a number parameter `b`.
pub(super) fn add_tool() -> ToolSpec {
    ToolSpec(json!({"type": "function", "function": {
        "name": "add",
        "parameters": {"type": "object", "properties": {
            "a": {"type": "integer"}, "b": {"type": "number"}}}}}))
}

/// How the job of an `ending_engine` ends after its text.
pub(super) enum Ending {
    /// The executor returns this error from `decode`.
    Error(EngError),
    /// The executor reports a cancellation that nobody requested.
    Cancel,
}

/// An executor that decodes one byte of its text per `decode` call and then ends as told.
struct Ends {
    ids: VecDeque<u32>,
    ending: Option<Ending>,
}

impl ExecutorOps for Ends {
    type Session = ();

    fn history<'s>(&self, _st: &'s ()) -> &'s [u32] {
        &[]
    }

    fn reset(&mut self, _st: &mut ()) -> Result<(), EngError> {
        Ok(())
    }

    fn prefill(&mut self, _st: &mut (), _toks: &[u32]) -> Result<(), EngError> {
        Ok(())
    }

    fn decode(
        &mut self,
        _st: &mut (),
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        if let Some(id) = self.ids.pop_front() {
            return Ok(DecodeBatch {
                tokens: vec![id],
                text: generation.decoder.push(id)?,
                finish: None,
            });
        }
        match self.ending.take() {
            Some(Ending::Error(error)) => Err(error),
            _ => Ok(DecodeBatch {
                tokens: Vec::new(),
                text: String::new(),
                finish: Some(DecodeFinish::Cancelled),
            }),
        }
    }

    fn boundary(&mut self, _st: &mut ()) {}
}

/// Starts an engine whose first job decodes `text` one byte at a time and then ends as `ending` says.
pub(super) fn ending_engine(text: &str, ending: Ending) -> Engine {
    let ids: VecDeque<u32> = ids(text).into();
    spawn_engine(512, move || {
        Ok((
            Ends {
                ids,
                ending: Some(ending),
            },
            (),
        ))
    })
    .expect("spawn engine")
}

/// An executor that records the prompt it receives and samples the ids 0 and 1 from the sampler.
/// It never finishes on its own, so the output limit of the job or a stop string ends the reply.
/// It sleeps `delay` before each decode and counts the decodes.
struct Probe {
    prompt: Arc<Mutex<Vec<u32>>>,
    sampling: Arc<Mutex<Option<SamplingParams>>>,
    delay: Duration,
    decodes: Arc<AtomicUsize>,
}

impl ExecutorOps for Probe {
    type Session = Vec<u32>;

    fn history<'s>(&self, st: &'s Vec<u32>) -> &'s [u32] {
        st
    }

    fn reset(&mut self, st: &mut Vec<u32>) -> Result<(), EngError> {
        st.clear();
        Ok(())
    }

    fn prefill(&mut self, st: &mut Vec<u32>, toks: &[u32]) -> Result<(), EngError> {
        st.extend_from_slice(toks);
        self.prompt
            .lock()
            .expect("prompt lock")
            .extend_from_slice(toks);
        Ok(())
    }

    fn decode(
        &mut self,
        st: &mut Vec<u32>,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        std::thread::sleep(self.delay);
        self.decodes.fetch_add(1, Ordering::SeqCst);
        if generation.cancel.load(Ordering::SeqCst) {
            return Ok(DecodeBatch {
                tokens: Vec::new(),
                text: generation.decoder.finish()?,
                finish: Some(DecodeFinish::Cancelled),
            });
        }
        if generation.remaining_tokens() == Some(0) {
            return Ok(DecodeBatch {
                tokens: Vec::new(),
                text: generation.decoder.finish()?,
                finish: Some(DecodeFinish::Length),
            });
        }
        // The first decode records the settings the chat handler resolved for this job.
        let mut seen = self.sampling.lock().expect("sampling lock");
        if seen.is_none() {
            *seen = Some(generation.sampler.params());
        }
        drop(seen);
        let id = generation.sampler.sample(&[0.0, 1.0])?;
        st.push(id);
        generation.record_generated(1);
        let mut text = generation.decoder.push(id)?;
        let finish = (generation.remaining_tokens() == Some(0)).then_some(DecodeFinish::Length);
        if finish.is_some() {
            text.push_str(&generation.decoder.finish()?);
        }
        Ok(DecodeBatch {
            tokens: vec![id],
            text,
            finish,
        })
    }

    fn boundary(&mut self, _st: &mut Vec<u32>) {}
}

/// A probe engine and the values that its executor shares with the test.
pub(super) struct Probes {
    pub(super) engine: Engine,
    /// Every prefilled token.
    pub(super) prompt: Arc<Mutex<Vec<u32>>>,
    /// The sampling settings of the first job it decoded, if any.
    pub(super) sampling: Arc<Mutex<Option<SamplingParams>>>,
    /// The number of `decode` calls.
    pub(super) decodes: Arc<AtomicUsize>,
}

/// Starts an engine on the probe executor. `delay` is the sleep before each decode.
pub(super) fn probes(delay: Duration) -> Probes {
    probes_behind(delay, None)
}

/// Starts a probe engine whose load waits for the returned sender to drop, so submitted jobs queue.
pub(super) fn gated_probes() -> (Probes, std::sync::mpsc::Sender<()>) {
    let (gate, wait) = std::sync::mpsc::channel::<()>();
    (probes_behind(Duration::ZERO, Some(wait)), gate)
}

/// Starts a probe engine. A gate delays the load until the gate drops.
fn probes_behind(delay: Duration, gate: Option<std::sync::mpsc::Receiver<()>>) -> Probes {
    let prompt = Arc::new(Mutex::new(Vec::new()));
    let sampling = Arc::new(Mutex::new(None));
    let decodes = Arc::new(AtomicUsize::new(0));
    let probe = Probe {
        prompt: Arc::clone(&prompt),
        sampling: Arc::clone(&sampling),
        delay,
        decodes: Arc::clone(&decodes),
    };
    let engine = spawn_engine(512, move || {
        if let Some(wait) = gate {
            let _ = wait.recv();
        }
        Ok((probe, Vec::new()))
    })
    .expect("spawn engine");
    Probes {
        engine,
        prompt,
        sampling,
        decodes,
    }
}

/// Starts an engine on the probe executor without a delay. The second value collects every prefilled token.
pub(super) fn probe_engine() -> (Engine, Arc<Mutex<Vec<u32>>>) {
    let probes = probes(Duration::ZERO);
    (probes.engine, probes.prompt)
}

/// Starts an engine on the probe executor without a delay. Besides every prefilled token, it
/// shares the sampling settings of the first job the engine decodes, so an end-to-end test can
/// observe the settings the chat handler resolved.
/// Starts an engine on the probe executor without a delay, sharing everything the executor
/// records: every prefilled token, the sampling settings of the first decoded job, and the
/// decode count, so an end-to-end test can observe the settings the chat handler resolved.
pub(super) fn probe_engine_with_sampling() -> Probes {
    probes(Duration::ZERO)
}

/// Waits up to five seconds until `condition` holds.
pub(super) fn wait_until(what: &str, condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// A `POST` request with a JSON body and the test key.
pub(super) fn post_json(uri: &str, body: &Value) -> Request<Body> {
    post_bytes(uri, Some(API_KEY), body.to_string().into_bytes())
}

/// A `POST` request with raw body bytes. `key` becomes the bearer token when it is set.
pub(super) fn post_bytes(uri: &str, key: Option<&str>, body: Vec<u8>) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(key) = key {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    builder.body(Body::from(body)).expect("build request")
}

/// One server-sent event: the optional `event:` name and the joined `data:` lines.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct SseEvent {
    pub(super) name: Option<String>,
    pub(super) data: String,
}

/// Splits a complete server-sent event stream into its events. Comment lines are skipped.
pub(super) fn sse_events(bytes: &Bytes) -> Vec<SseEvent> {
    let text = std::str::from_utf8(bytes).expect("utf-8 stream");
    text.split("\n\n")
        .filter(|block| !block.is_empty())
        .filter_map(|block| {
            let mut name = None;
            let mut data = Vec::new();
            for line in block.lines() {
                if let Some(value) = line.strip_prefix("event: ") {
                    name = Some(value.to_string());
                } else if let Some(value) = line.strip_prefix("data: ") {
                    data.push(value);
                } else if !line.starts_with(':') {
                    panic!("unexpected line in an event stream: {line:?}");
                }
            }
            (name.is_some() || !data.is_empty()).then(|| SseEvent {
                name,
                data: data.join("\n"),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        EOS, SseEvent, bundle, gated_engine, generation, ids, probes, refusing_engine, script,
        sse_events, wait_for_state,
    };
    use crate::EngError;
    use crate::family::executor::GenerationState;
    use crate::family::sampler::{Sampler, SamplingParams};
    use crate::sched::job::{Job, JobEvent};
    use crate::sched::worker::EngineState;
    use crate::tokenizer::{DecodeFinish, IncrementalDecoder};
    use axum::body::Bytes;
    use std::time::Duration;

    #[test]
    fn the_fixture_tokenizer_encodes_ascii_as_byte_values() {
        let bundle = bundle();
        assert_eq!(
            bundle.tokenizer().encode("hi!").expect("encode"),
            ids("hi!")
        );
        assert_eq!(bundle.eos_token_id(), EOS);
        assert_eq!(bundle.vocab_size(), 258);
    }

    #[test]
    fn a_script_has_one_batch_per_byte_and_ends_with_eos() {
        assert_eq!(script("ab"), vec![vec![97], vec![98], vec![EOS]]);
    }

    #[test]
    fn a_gated_engine_loads_only_after_the_gate_drops() {
        let (engine, gate) = gated_engine(script("a"));
        let handle = engine.handle();
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(handle.state(), EngineState::Loading);
        drop(gate);
        wait_for_state(&handle, EngineState::Ready);
    }

    #[test]
    fn a_refusing_engine_fails_the_first_job_with_its_error() {
        let engine = refusing_engine(EngError::MemoryPlan("no room".to_string()));
        let (job, mut handle) = Job::new(vec![104], generation()).expect("job");
        engine.handle().try_submit(job).expect("submit");
        let event = handle.events.blocking_recv().expect("an event");
        assert_eq!(
            event,
            JobEvent::Error(EngError::MemoryPlan("no room".to_string()))
        );
    }

    #[test]
    fn the_probe_engine_records_the_prompt_and_samples_only_ids_zero_and_one() {
        let probes = probes(Duration::ZERO);
        let params = SamplingParams::new(1.0, 1.0, 0).expect("params");
        let decoder = IncrementalDecoder::new(bundle().tokenizer());
        let generation = GenerationState::with_sampler(decoder, Sampler::new(params, 7));
        let (job, mut handle) = Job::new(vec![104, 105], generation).expect("job");
        probes.engine.handle().try_submit(job).expect("submit");
        let mut ids = std::collections::BTreeSet::new();
        for _ in 0..40 {
            let Some(JobEvent::Batch(batch)) = handle.events.blocking_recv() else {
                panic!("expected a batch");
            };
            ids.extend(batch.tokens);
        }
        handle.cancel();
        assert_eq!(ids, std::collections::BTreeSet::from([0, 1]));
        assert_eq!(*probes.prompt.lock().expect("prompt lock"), vec![104, 105]);
    }

    #[test]
    fn the_probe_engine_ends_a_job_at_its_output_limit() {
        let probes = probes(Duration::ZERO);
        let params = SamplingParams::new(1.0, 1.0, 0).expect("params");
        let decoder = IncrementalDecoder::new(bundle().tokenizer());
        let generation =
            GenerationState::with_sampler(decoder, Sampler::new(params, 7)).with_max_tokens(3);
        let (job, mut handle) = Job::new(vec![104], generation).expect("job");
        probes.engine.handle().try_submit(job).expect("submit");
        let mut finishes = Vec::new();
        while finishes.len() < 5 {
            let Some(JobEvent::Batch(batch)) = handle.events.blocking_recv() else {
                break;
            };
            finishes.push((batch.tokens.len(), batch.finish));
            if batch.finish.is_some() {
                break;
            }
        }
        assert_eq!(
            finishes,
            vec![(1, None), (1, None), (1, Some(DecodeFinish::Length))]
        );
    }

    #[test]
    fn sse_events_splits_named_and_plain_events_and_skips_comments() {
        let stream = Bytes::from_static(
            b": keep-alive\n\ndata: {\"a\":1}\n\nevent: error\ndata: {\"b\":2}\n\ndata: [DONE]\n\n",
        );
        assert_eq!(
            sse_events(&stream),
            vec![
                SseEvent {
                    name: None,
                    data: "{\"a\":1}".to_string()
                },
                SseEvent {
                    name: Some("error".to_string()),
                    data: "{\"b\":2}".to_string()
                },
                SseEvent {
                    name: None,
                    data: "[DONE]".to_string()
                },
            ]
        );
    }
}
