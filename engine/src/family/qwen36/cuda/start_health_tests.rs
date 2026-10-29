//! The health pins of the CUDA serve start and its ignored device test. The health pins need no
//! device: a real engine worker whose executor fails with the terminal-latch error of
//! `kernels::cuda::ensure_usable`, and one whose load fails, pin the worker's own rules —
//! `Backend` is process-scoped (`NativeFailed`, queued jobs rejected, no hang), a load failure
//! leaves `Error` with the message, `Loading` until `make` returns. The mapping needs no
//! production code: these pins hold it in place. The device test is ignored and runs only on
//! the 3090, held exclusively.

use crate::EngError;
use crate::exec_fake::FakeExecutor;
use crate::family::executor::{ExecutorOps, GenerationState};
use crate::family::qwen36::factory;
use crate::sched::job::{Job, JobEvent};
use crate::sched::worker::{EngineState, SubmitError, spawn_engine};
use crate::tokenizer::{DecodeBatch, DecodeFinish, IncrementalDecoder, Tokenizer};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The three-token tokenizer of the worker tests; the ids, not the text, are what is checked.
const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2},"merges":[]}}"#;

/// The end-of-sequence id of the fake scripts (outside the three-token vocabulary).
const EOS: u32 = 9;

fn generation() -> GenerationState {
    let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).expect("parse tokenizer");
    GenerationState::new(IncrementalDecoder::new(&tokenizer))
}

fn job(prompt: &[u32], max_tokens: u32) -> (Job, crate::sched::job::JobHandle) {
    Job::new(prompt.to_vec(), generation().with_max_tokens(max_tokens)).expect("job")
}

/// Spins until the condition holds. It fails the test after ten seconds instead of hanging.
fn wait_until(what: &str, condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::yield_now();
    }
}

/// One job event with a wall-clock cap: tokio's receiver has no blocking `recv_timeout`, so the
/// wait parks on the shared current-thread runtime under `timeout_at` — no loop of the device
/// test waits without a deadline. `None` (the channel closed) passes through; only the clock
/// panics.
fn event_before<T>(
    runtime: &tokio::runtime::Runtime,
    events: &mut tokio::sync::mpsc::Receiver<T>,
    deadline: Instant,
    what: &str,
) -> Option<T> {
    // The timer future must be created INSIDE the runtime (an async block): `timeout_at`
    // registers with the runtime's time driver when it is built, and outside `block_on` there
    // is no reactor ("there is no reactor running", the first 3090 run of this helper).
    runtime
        .block_on(async {
            tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), events.recv()).await
        })
        .unwrap_or_else(|_| panic!("timed out waiting for {what}"))
}

/// The shared current-thread runtime of the device test's bounded waits.
fn event_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("current-thread runtime")
}

/// An executor whose `decode` fails with the exact terminal-latch error that
/// `kernels::cuda::ensure_usable` returns after a native failure (`Backend`, `execute`):
/// the same variant, operation and message; the pins hold the WORKER's mapping.
struct TerminalDecode {
    inner: FakeExecutor,
}

impl ExecutorOps for TerminalDecode {
    type Session = <FakeExecutor as ExecutorOps>::Session;

    fn history<'s>(&self, st: &'s Self::Session) -> &'s [u32] {
        self.inner.history(st)
    }

    fn reset(&mut self, st: &mut Self::Session) -> Result<(), EngError> {
        self.inner.reset(st)
    }

    fn prefill(&mut self, st: &mut Self::Session, toks: &[u32]) -> Result<(), EngError> {
        self.inner.prefill(st, toks)
    }

    fn decode(
        &mut self,
        _st: &mut Self::Session,
        _generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        Err(EngError::Backend {
            operation: "execute",
            message: "CUDA backend is terminal after an earlier native failure".to_string(),
        })
    }

    fn boundary(&mut self, st: &mut Self::Session) {
        self.inner.boundary(st);
    }
}

#[test]
fn a_terminal_cuda_failure_is_process_scoped_for_the_worker() {
    // Requirement 4, the worker's rule: ANY Backend error ends the engine (NativeFailed),
    // the queued jobs are rejected, the late submit is refused, nothing hangs.
    let engine = spawn_engine(8, || {
        let inner = FakeExecutor::new(vec![vec![0], vec![0], vec![EOS]], &[EOS], 100_000);
        Ok((
            TerminalDecode { inner },
            <FakeExecutor as ExecutorOps>::Session::default(),
        ))
    })
    .expect("spawn");
    let handle = engine.handle();
    let (running, mut running_events) = job(&[2], 4);
    let (queued, mut queued_events) = job(&[2], 4);
    handle.try_submit(running).expect("submit running");
    handle.try_submit(queued).expect("submit queued");
    let error = EngError::Backend {
        operation: "execute",
        message: "CUDA backend is terminal after an earlier native failure".to_string(),
    };
    let mut events = Vec::new();
    while let Some(event) = running_events.events.blocking_recv() {
        events.push(event);
    }
    assert_eq!(
        events,
        vec![JobEvent::Error(error)],
        "the running job sees the latch error"
    );
    let mut seen = Vec::new();
    while let Some(event) = queued_events.events.blocking_recv() {
        seen.push(event);
    }
    assert_eq!(
        seen.len(),
        1,
        "the queued job is rejected once, not hung: {seen:?}"
    );
    assert!(matches!(seen[0], JobEvent::Error(EngError::Backend { .. })));
    wait_until("the engine to end in NativeFailed", || {
        handle.state() == EngineState::NativeFailed
    });
    let (late, _late_events) = job(&[2], 4);
    assert_eq!(handle.try_submit(late), Err(SubmitError::NativeFailed));
    drop(handle);
    assert_eq!(engine.shutdown().state, EngineState::NativeFailed);
}

#[test]
fn a_load_failure_leaves_the_engine_in_error_with_the_message() {
    // `Loading` until `make` returns; a typed load failure ends in `Error` with the message
    // and the queued jobs are rejected with LoadFailed.
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let engine = spawn_engine::<FakeExecutor, _>(16, move || {
        entered_tx.send(()).expect("entered make");
        let _ = release_rx.recv();
        Err(EngError::MemoryPlan(
            "cuda admission: 23 bytes free after the load is below the 1073741824-byte floor"
                .to_string(),
        ))
    })
    .expect("spawn");
    let handle = engine.handle();
    assert_eq!(handle.state(), EngineState::Loading, "before make returns");
    entered_rx.recv().expect("make is running");
    assert_eq!(handle.state(), EngineState::Loading, "while make runs");
    let (queued, mut queued_events) = job(&[2], 4);
    handle.try_submit(queued).expect("submit queued");
    release_tx.send(()).expect("release make");
    wait_until("the engine to end in Error", || {
        handle.state() == EngineState::Error
    });
    let failure = handle.failure().expect("the load failure is recorded");
    assert!(failure.contains("cuda admission"), "{failure}");
    let mut seen = Vec::new();
    while let Some(event) = queued_events.events.blocking_recv() {
        seen.push(event);
    }
    assert_eq!(
        seen.len(),
        1,
        "the queued job is rejected once, not hung: {seen:?}"
    );
    assert!(matches!(seen[0], JobEvent::Error(_)));
    let (late, _late_events) = job(&[2], 4);
    assert_eq!(handle.try_submit(late), Err(SubmitError::LoadFailed));
    drop(handle);
    assert_eq!(engine.shutdown().state, EngineState::Error);
}

/// The reference chain of one free-run: the per-step argmax ids (the device chain the gate
/// feeds) and the top values of each step's row (for the divergence report).
struct Reference {
    ids: Vec<u32>,
    top: Vec<Vec<(u32, f32)>>,
}

/// The reference of the comparison: a `cuda-rd-run` raw file (`prompts[].runs[].steps`,
/// top32) or an oracle `run.json` (`steps`, top 5), detected by the shape.
fn read_reference(path: &str) -> Reference {
    let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("{path}: {error}"));
    let value: serde_json::Value = serde_json::from_str(&text).expect("the reference json");
    let steps = value
        .pointer("/prompts/0/runs/0/steps")
        .or_else(|| value.get("steps"))
        .expect("the reference has a steps list (raw or oracle shape)");
    let mut ids = Vec::with_capacity(steps.as_array().map_or(0, Vec::len));
    let mut top = Vec::new();
    for step in steps.as_array().expect("steps is a list") {
        ids.push(step["argmax"].as_u64().expect("argmax") as u32);
        // The raw shape names the list `top32`, the oracle shape `top`.
        let pairs = step
            .get("top32")
            .or_else(|| step.get("top"))
            .and_then(serde_json::Value::as_array)
            .expect("top32 (raw) or top (oracle)")
            .iter()
            .map(|pair| {
                (
                    pair[0].as_u64().expect("top id") as u32,
                    pair[1].as_f64().expect("top logit") as f32,
                )
            })
            .collect();
        top.push(pairs);
    }
    Reference { ids, top }
}

/// P5's prompt ids from the qualification fixtures (the oracle's own input).
fn p5_ids() -> Vec<u32> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../qualification/cuda-rd/oracle/prompts/P5.ids.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let value: serde_json::Value = serde_json::from_str(&text).expect("P5 ids json");
    value["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|id| id.as_u64().expect("id") as u32)
        .collect()
}

#[test]
#[ignore = "the 3090, held exclusively: starts the real engine"]
fn cuda_engine_streams_the_gate_tokens() {
    // Tie-break rules, from the sources. HOST (family/sampler.rs `argmax`): greedy draws the
    // LOWEST index among the largest logits (`value > best_value` replaces, so an equal value
    // never wins). DEVICE (kernels/cuda/resident/step_params.rs `argmax`): the largest value
    // wins, the LOWEST index on a tie, a NaN never wins. The rules agree on finite rows, so a
    // divergence is numeric — unless the reference top two logits are EXACTLY equal at the
    // divergence, which the report names as a rule difference.
    let model_dir = std::env::var("Q36_CUDA_MODEL_DIR")
        .unwrap_or_else(|_| "/media/dan/nvme/models/qwen3.6-35b-a3b".to_string());
    let tokenizer_dir =
        std::env::var("Q36_CUDA_TOKENIZER_DIR").unwrap_or_else(|_| model_dir.clone());
    let reference_path = std::env::var("Q36_CUDA_REFERENCE").unwrap_or_else(|_| {
        format!(
            "{}/../qualification/cuda-rd/oracle/cpu/clean/P5/run.json",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    // A 1,024-position context: P5 (612 prompt tokens) plus 128 generated fit, and the overflow
    // case below prefills at most 1,024 tokens before the session refuses it (the session
    // checks the running total per prompt chunk, so the default 32,768 would take ~20 min on
    // the tiled step).
    let options = crate::family::qwen36::settings::Q36Options {
        context_limit: Some(1024),
        ..Default::default()
    };
    let began = Instant::now();
    let started = factory::start_engine(
        std::path::Path::new(&model_dir),
        std::path::Path::new(&tokenizer_dir),
        Some(&options),
        false,
        None,
    )
    .expect("the engine starts through the factory arm (one engine per process)");
    let handle = started.engine.handle();
    // Jobs of the REAL engine decode with the engine's own pinned bundle: the toy tokenizer of
    // the fake-engine helper `job` has one token and refuses every real id.
    let real_job = |prompt: &[u32], max_tokens: u32| {
        let decoder = IncrementalDecoder::new(started.bundle.tokenizer());
        Job::new(
            prompt.to_vec(),
            GenerationState::new(decoder).with_max_tokens(max_tokens),
        )
        .expect("job")
    };
    // The real load takes 15-60 s (20 GB through the page cache): the 10 s helper of the fake
    // engines is too short. A load failure ends the wait at once with the recorded cause.
    let deadline = Instant::now() + Duration::from_secs(240);
    loop {
        let state = handle.state();
        if state == EngineState::Ready {
            break;
        }
        assert!(
            !matches!(state, EngineState::Error | EngineState::NativeFailed),
            "the engine failed to start: {state:?}, {:?}",
            handle.failure()
        );
        assert!(
            Instant::now() < deadline,
            "timed out (240 s) waiting for Ready: {state:?}"
        );
        std::thread::sleep(Duration::from_millis(100));
    }

    eprintln!(
        "{:.1} s: the engine is Ready",
        began.elapsed().as_secs_f64()
    );
    // Stream P5 greedily: the default job sampler IS greedy (temperature 0), so every id
    // comes from the host sampler's draw over the f32 logits row the executor reads back.
    let prompt = p5_ids();
    let reference = read_reference(&reference_path);
    let runtime = event_runtime();
    let (stream, mut stream_events) = real_job(&prompt, 128);
    handle.try_submit(stream).expect("submit the stream job");
    let mut streamed = Vec::new();
    let mut finish = None;
    let deadline = Instant::now() + Duration::from_secs(600);
    while let Some(event) = event_before(
        &runtime,
        &mut stream_events.events,
        deadline,
        "the stream job's first event",
    ) {
        match event {
            JobEvent::Batch(batch) => {
                streamed.extend_from_slice(&batch.tokens);
                if let Some(reason) = batch.finish {
                    finish = Some(reason);
                    break;
                }
            }
            JobEvent::Finished(_) => break,
            JobEvent::Error(error) => panic!("the stream job failed: {error:?}"),
        }
    }
    // The terminal event of a job can arrive before the worker leaves Busy: wait for Ready
    // (a failed engine never gets there and the helper times out).
    wait_until("the engine to be Ready after the stream", || {
        handle.state() == EngineState::Ready
    });
    // Compare with the reference chain: the first divergence names the position, both ids and
    // the logit gap from the reference's own top list, or reports an exact tie as such.
    // The whole chain, up front: the P5 oracle has 128 steps and no EOG, so the reference must
    // carry at least 128 and the stream must cover it EXACTLY — an early finish (or a short
    // reference file) is itself a divergence, not an excuse.
    assert!(
        reference.ids.len() >= 128,
        "the reference chain is too short: {} steps",
        reference.ids.len()
    );
    assert_eq!(
        streamed.len(),
        reference.ids.len(),
        "the stream ended early: finish {finish:?}"
    );
    for (position, (got, want)) in streamed.iter().zip(&reference.ids).enumerate() {
        let (got, want) = (*got, *want);
        if got != want {
            let top = &reference.top[position];
            let logit = |id: u32| top.iter().find(|p| p.0 == id).map(|p| p.1);
            let gap = match (logit(got), logit(want)) {
                (Some(g), Some(w)) => {
                    format!("logit gap {:.6} (ours {g:?}, reference {w:?})", g - w)
                }
                (Some(g), None) => format!("ours at {g:?}, the reference id outside the top list"),
                (None, Some(w)) => format!("ours outside the top list, the reference id at {w:?}"),
                (None, None) => "neither id is in the reference top list".to_string(),
            };
            if top.len() >= 2 && top[0].1 == top[1].1 {
                panic!(
                    "divergence at {position}: ours {got}, reference {want}; the reference top \
                     two logits are EXACTLY tied ({:?}, {:?}) — a tie-break rule difference",
                    top[0], top[1]
                );
            }
            panic!("divergence at {position}: ours {got}, reference {want}; {gap}");
        }
    }

    eprintln!(
        "{:.1} s: P5 streamed and compared",
        began.elapsed().as_secs_f64()
    );
    // Cancel mid-decode: a long job stops after its current piece, the next job still runs.
    let (long_job, long_handle) = real_job(&prompt, 100_000);
    handle.try_submit(long_job).expect("submit the long job");
    let mut long_events = long_handle;
    let mut asked = false;
    let mut long_finish = None;
    let deadline = Instant::now() + Duration::from_secs(300);
    while let Some(event) =
        event_before(&runtime, &mut long_events.events, deadline, "the long job")
    {
        match event {
            JobEvent::Batch(batch) => {
                if let Some(reason) = batch.finish {
                    long_finish = Some(reason);
                    break;
                }
                if !asked {
                    long_events.cancel();
                    asked = true;
                }
            }
            JobEvent::Finished(_) => break,
            // An error is a FAILURE, not an end.
            JobEvent::Error(error) => panic!("the long job failed: {error:?}"),
        }
    }
    // The long job ended BY THE CANCEL: its terminal batch says so.
    assert_eq!(
        long_finish,
        Some(DecodeFinish::Cancelled),
        "the long job did not end by the cancel"
    );
    wait_until("the engine to be Ready again", || {
        handle.state() == EngineState::Ready
    });
    let (next, mut next_events) = real_job(&[2], 2);
    handle.try_submit(next).expect("submit after the cancel");
    let mut next_tokens = 0;
    let deadline = Instant::now() + Duration::from_secs(120);
    while let Some(event) =
        event_before(&runtime, &mut next_events.events, deadline, "the next job")
    {
        match event {
            JobEvent::Batch(batch) => next_tokens += batch.tokens.len(),
            JobEvent::Finished(_) => break,
            // "The next request works" is ASSERTED, not assumed: an error fails the test.
            JobEvent::Error(error) => panic!("the next job failed: {error:?}"),
        }
    }
    assert!(next_tokens >= 1, "the next job produced no tokens");
    wait_until("the engine to survive the cancel (Ready)", || {
        handle.state() == EngineState::Ready
    });

    eprintln!(
        "{:.1} s: the cancel and the next job are done",
        began.elapsed().as_secs_f64()
    );
    // A prompt at the context limit is refused cleanly: the typed over-length form, no
    // panic, Ready.
    let limit = usize::try_from(started.context_limit).expect("the context limit");
    let overflow: Vec<u32> = vec![198; limit];
    let (over, mut over_events) = real_job(&overflow, 4);
    handle.try_submit(over).expect("submit the overflow job");
    let mut refused = None;
    let deadline = Instant::now() + Duration::from_secs(120);
    while let Some(event) = event_before(
        &runtime,
        &mut over_events.events,
        deadline,
        "the overflow job",
    ) {
        if let JobEvent::Error(error) = event {
            refused = Some(error);
            break;
        }
    }
    match refused {
        Some(
            error @ EngError::PromptTooLong {
                prompt_tokens,
                limit_tokens,
            },
        ) => {
            assert_eq!(prompt_tokens, limit as u64, "{error:?}");
            assert_eq!(limit_tokens, limit as u64 - 1, "{error:?}");
        }
        other => panic!("expected a typed over-length refusal, got {other:?}"),
    }
    wait_until("the engine to be Ready after the overflow", || {
        handle.state() == EngineState::Ready
    });
    drop(handle);
    let report = started.engine.shutdown();
    assert_eq!(report.state, EngineState::Stopped, "{report:?}");
}
