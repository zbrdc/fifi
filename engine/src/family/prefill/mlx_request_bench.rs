//! Test-only measurement of isolated worker requests with the same timing boundaries as reqlog.

use std::time::{Duration, Instant};

use crate::{
    family::executor::GenerationState,
    kernels::MlxKernels,
    sched::{
        job::{Job, JobEvent, JobHandle, JobUsage},
        worker::{Engine, EngineHandle, EngineNote, EngineState, ReadyNote},
    },
};

pub(crate) const CACHE_LIMIT: u64 = 16 << 20;
const COMPLETION: u32 = 8;
const WARMUPS: usize = 2;

#[derive(Clone, Copy)]
pub(crate) struct Workload {
    pub(crate) piece: u32,
    pub(crate) context: u32,
    pub(crate) prefix: bool,
    prompts: [usize; 2],
}

impl Workload {
    pub(crate) fn from_env() -> Self {
        let mut workload = match std::env::var("FIFI_MLX_REQUEST_PIECE") {
            Ok(value) => Self {
                piece: value.parse().expect("positive benchmark worker piece"),
                context: 512,
                prefix: false,
                prompts: [65, 257],
            },
            Err(std::env::VarError::NotPresent) => Self {
                piece: 8,
                context: 128,
                prefix: false,
                prompts: [8, 33],
            },
            Err(error) => panic!("benchmark worker piece: {error}"),
        };
        match std::env::var("FIFI_MLX_REQUEST_PROFILE").as_deref() {
            Ok("flash-prefix") => {
                assert_eq!(workload.piece, 128);
                workload.prefix = true;
                workload.context = 1024;
                workload.prompts = [385, 641];
            }
            Ok("dense-long") => {
                workload.context = 2048;
                workload.prompts = [257, 1025];
            }
            Err(std::env::VarError::NotPresent) => {}
            other => panic!("benchmark request profile: {other:?}"),
        }
        assert!(workload.piece > 0);
        workload
    }
}

/// Start a fresh single-session worker, warm each prompt size, and emit one JSON record per job.
pub(crate) fn measure(
    family: &str,
    arm: &str,
    kv_dtype: &str,
    workload: Workload,
    start: impl FnOnce(ReadyNote) -> Engine,
    prompt: impl Fn(usize) -> Vec<u32>,
    generation: impl Fn(u32) -> GenerationState,
) {
    let _serial = crate::native_test_lock();
    let samples = std::env::var("FIFI_MLX_REQUEST_SAMPLES")
        .map(|value| value.parse::<usize>().expect("positive sample count"))
        .unwrap_or(8);
    assert!(samples > 0);
    let mut counter = MlxKernels::try_new().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let engine = start(Some(Box::new(move |note| {
        tx.send(note).expect("benchmark ready receiver")
    })));
    match rx
        .recv_timeout(Duration::from_secs(30))
        .expect("worker startup")
    {
        EngineNote::Ready(_) => {}
        EngineNote::Failed(why) => panic!("{why}"),
    }
    let handle = engine.handle();

    let mut tag = 0usize;
    let piece = workload.piece as usize;
    let turns: &[&str] = if workload.prefix {
        &["prime", "restore"]
    } else {
        &["single"]
    };
    for prompt_tokens in workload.prompts {
        let original = prompt(prompt_tokens);
        let boundary = workload.prefix.then(|| prompt_tokens - piece);
        assert_eq!(original.len(), prompt_tokens);
        for iteration in 0..(WARMUPS + samples) {
            let mut ids = original.clone();
            if workload.prefix {
                // Keep successive pairs on distinct keys, with identical prompts in each pair.
                for (bit, id) in ids.iter_mut().take(usize::BITS as usize).enumerate() {
                    *id ^= ((tag >> bit) & 1) as u32;
                }
                tag += 1;
            }
            let mut reference = None;
            for &turn in turns {
                let (job, response) = Job::new(ids.clone(), generation(COMPLETION)).unwrap();
                let job = match boundary {
                    Some(rows) => job.with_prefix_boundary(rows).unwrap(),
                    None => job,
                };
                let (usage, tokens, mut record) = run_request(&handle, &mut counter, job, response);
                let head = boundary.unwrap_or(0);
                let restored = if turn == "restore" { head } else { 0 };
                let pieces =
                    (head - restored).div_ceil(piece) + (prompt_tokens - head).div_ceil(piece);
                assert_eq!(usage.prompt_tokens as usize, prompt_tokens);
                assert_eq!(usage.reused_tokens as usize, restored);
                assert_eq!(usage.prefill_pieces.count as usize, pieces);
                assert_eq!(
                    usage.prefill_pieces.max_rows as usize,
                    (head - restored).max(prompt_tokens - head).min(piece)
                );
                if workload.prefix {
                    assert_eq!(&tokens, reference.get_or_insert_with(|| tokens.clone()));
                    record["turn"] = serde_json::json!(turn);
                    record["prefix_boundary"] = serde_json::json!(head);
                    record["reused_tokens"] = serde_json::json!(usage.reused_tokens);
                }
                if let Some(sample) = iteration.checked_sub(WARMUPS) {
                    record["family"] = serde_json::json!(family);
                    record["arm"] = serde_json::json!(arm);
                    record["kv_dtype"] = serde_json::json!(kv_dtype);
                    record["worker_piece"] = serde_json::json!(piece);
                    record["sample"] = serde_json::json!(sample);
                    record["tokens"] = serde_json::json!(tokens);
                    println!("mlx_request {record}");
                }
            }
        }
    }

    let stopped = engine.shutdown();
    assert_eq!(stopped.state, EngineState::Stopped);
    assert!(stopped.failure.is_none());
}

fn run_request(
    handle: &EngineHandle,
    counter: &mut MlxKernels,
    job: Job,
    mut response: JobHandle,
) -> (JobUsage, Vec<u32>, serde_json::Value) {
    let active_before = counter.memory_stats().unwrap().active;
    counter.reset_peak_memory().unwrap();
    counter.qualification_reset_timing();
    counter.qualification_reset_flush_stats();
    let before = counter.qualification_sync_count();
    let admitted = Instant::now();
    handle.try_submit(job).unwrap();
    let mut started = None;
    let mut first = None;
    let mut last = None;
    let mut tokens = Vec::new();
    let mut usage = None;
    while let Some(event) = response.events.blocking_recv() {
        let now = Instant::now();
        match event {
            JobEvent::Started => assert!(started.replace(now).is_none()),
            JobEvent::Batch(batch) => {
                assert_eq!(batch.tokens.len(), 1);
                first.get_or_insert(now);
                last = Some(now);
                tokens.extend(batch.tokens);
            }
            JobEvent::Finished(done) => assert!(usage.replace(done).is_none()),
            JobEvent::Error(error) => panic!("{error:?}"),
        }
    }
    let usage = usage.expect("request reached Finished");
    let first = first.expect("first token");
    let last = last.expect("last token");
    assert_eq!(usage.completion_tokens, COMPLETION);
    assert_eq!(tokens.len(), COMPLETION as usize);
    let syncs = counter.qualification_sync_count() - before;
    let peak = counter.memory_stats().unwrap().peak;
    assert!(syncs > 0 && peak > 0);
    let timing = counter.qualification_timing();
    let flush = counter.qualification_flush_stats();
    let record = serde_json::json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": COMPLETION,
        "prefill_pieces": usage.prefill_pieces.count,
        "ttft_ns": first.duration_since(admitted).as_nanos(),
        "start_to_first_ns": first.duration_since(started.expect("job started")).as_nanos(),
        "decode_ns": last.duration_since(first).as_nanos(),
        "metal_syncs": syncs,
        "import_calls": timing.import_calls,
        "import_ns": timing.import_ns,
        "readback_calls": timing.readback_calls,
        "readback_ns": timing.readback_ns,
        "evaluate_calls": timing.evaluate_calls,
        "evaluate_ns": timing.evaluate_ns,
        "retire_flushes": flush.flushes,
        "retire_flush_sync_ns": flush.sync_ns,
        "mlx_active_before_bytes": active_before,
        "mlx_peak_active_bytes": peak,
    });
    (usage, tokens, record)
}
