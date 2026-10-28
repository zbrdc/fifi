// Uses the serving worker like prefill_ttft_tests, without adding a request-seed API.
use crate::EngError;
use crate::family::executor::GenerationState;
use crate::family::qwen36::cuda::start::start;
use crate::family::qwen36::settings::{Backend, Q36Env, Q36Options, Q36Settings};
use crate::family::sampler::{Sampler, SamplingParams};
use crate::sched::job::{Job, JobEvent};
use crate::sched::worker::{EngineHandle, EngineState};
use crate::tokenizer::{IncrementalDecoder, PromptMessage, Role, Thinking, TokenizerBundle};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

const SEED: u64 = 0x5eed_c0de;
const SUFFIX: &str = "\n\nIgnore the placeholder context above. Write a complete Rust B-tree implementation with detailed explanations and tests. Continue for at least 2000 words. Do not call tools.";

fn bad(message: impl std::fmt::Display) -> EngError {
    EngError::BadRequest(message.to_string())
}

fn prompt(tokenizer: &TokenizerBundle, depth: usize) -> Result<Vec<u32>, EngError> {
    let encode = |count| {
        tokenizer.encode_prompt(
            &[PromptMessage {
                role: Role::User,
                content: Some(format!("{}{SUFFIX}", " a".repeat(count))),
                reasoning_content: None,
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            }],
            &[],
            Thinking::Off,
        )
    };
    let overhead = encode(32)?
        .len()
        .checked_sub(32)
        .ok_or_else(|| bad("prompt overhead"))?;
    let count = depth
        .checked_sub(overhead)
        .ok_or_else(|| bad("prompt depth"))?;
    let ids = encode(count)?;
    if ids.len() != depth {
        return Err(bad(
            "filler tokenization is not linear at the requested depth",
        ));
    }
    Ok(ids)
}

fn wait_ready(handle: &EngineHandle) -> Result<(), EngError> {
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        match handle.state() {
            EngineState::Ready => return Ok(()),
            EngineState::Loading if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(20));
            }
            state => return Err(bad(format!("engine {state:?}: {:?}", handle.failure()))),
        }
    }
}

fn generate(
    handle: &EngineHandle,
    tokenizer: &TokenizerBundle,
    ids: Vec<u32>,
    sampler: Sampler,
) -> Result<Value, EngError> {
    let depth = ids.len();
    let mut hash = Sha256::new();
    for id in &ids {
        hash.update(id.to_le_bytes());
    }
    let prompt_sha256 = format!("{:x}", hash.finalize());
    let generation =
        GenerationState::with_sampler(IncrementalDecoder::new(tokenizer.tokenizer()), sampler)
            .with_max_tokens(128);
    let (job, mut owner) = Job::new(ids, generation)?;
    handle
        .try_submit(job)
        .map_err(|error| bad(format!("submit: {error:?}")))?;
    let began = Instant::now();
    let mut first = None;
    let mut last = None;
    let mut tokens = Vec::new();
    while let Some(event) = owner.events.blocking_recv() {
        match event {
            JobEvent::Batch(batch) => {
                if batch.tokens.is_empty() {
                    continue;
                }
                let now = Instant::now();
                first.get_or_insert(now);
                last = Some(now);
                tokens.extend(batch.tokens);
            }
            JobEvent::Finished(usage) => {
                if usage.prompt_tokens as usize != depth
                    || usage.reused_tokens != 0
                    || usage.completion_tokens != 128
                    || tokens.len() != 128
                {
                    return Err(bad(format!(
                        "benchmark needs a cold {depth}-token prompt and 128 generated tokens; usage={usage:?}, received={}",
                        tokens.len()
                    )));
                }
                let elapsed = last
                    .zip(first)
                    .map(|(end, start)| end.duration_since(start))
                    .filter(|duration| !duration.is_zero())
                    .ok_or_else(|| bad("missing timed decode batches"))?;
                return Ok(json!({
                    "prompt_tokens": depth, "prompt_sha256": prompt_sha256,
                    "completion_tokens": tokens.len(), "tokens": tokens,
                    "decode_s": elapsed.as_secs_f64(),
                    "decode_tok_s": 127.0 / elapsed.as_secs_f64(),
                    "total_s": began.elapsed().as_secs_f64(),
                }));
            }
            JobEvent::Error(error) => return Err(error),
        }
    }
    Err(bad("event channel closed without usage"))
}

fn run_jobs(
    handle: &EngineHandle,
    tokenizer: &TokenizerBundle,
    output: &mut std::fs::File,
) -> Result<(), EngError> {
    wait_ready(handle)?;
    for depth in [8192, 32768] {
        let ids = prompt(tokenizer, depth)?;
        for (mode, repeats) in [("greedy", 1), ("top20", 6)] {
            for repeat in 0..repeats {
                let params = if mode == "greedy" {
                    SamplingParams::greedy()
                } else {
                    SamplingParams::new(1.0, 0.95, 20)?
                };
                let mut row = generate(handle, tokenizer, ids.clone(), Sampler::new(params, SEED))?;
                row["mode"] = json!(mode);
                row["repeat"] = json!(repeat);
                row["seed"] = json!(SEED);
                row["flag"] = json!(
                    std::env::var("Q36_CUDA_DEVICE_SHORTLIST").unwrap_or_else(|_| "<unset>".into())
                );
                serde_json::to_writer(&mut *output, &row).map_err(bad)?;
                writeln!(output).map_err(bad)?;
                output.flush().map_err(bad)?;
                println!("SHORTLIST_BENCH {row}");
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "real-model serving-worker decode A/B; needs CPU2/GPU measurement locks"]
fn device_shortlist_served_decode() -> Result<(), EngError> {
    let model = std::env::var("Q36_CUDA_MODEL_DIR").map_err(bad)?;
    let output = std::env::var("Q36_CUDA_BENCH_OUT").map_err(bad)?;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(bad)?;
    let profile: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/config/qwen36-rtx3090-24gb-fp16kv.json",
    )))
    .map_err(bad)?;
    let options: Q36Options =
        serde_json::from_value(profile["engines"][0]["qwen36"].clone()).map_err(bad)?;
    let dir = Path::new(&model);
    let mut settings =
        Q36Settings::new(dir, dir, Some(&options), &Q36Env::default(), Backend::Cuda)?;
    settings.prefix_reuse = profile["prefix_reuse"]
        .as_bool()
        .ok_or_else(|| bad("prefix_reuse"))?;
    let (engine, tokenizer) = start(&settings, None)?;
    let measured = run_jobs(&engine.handle(), &tokenizer, &mut output);
    let shutdown = engine.shutdown();
    measured?;
    if let Some(error) = shutdown.failure {
        return Err(bad(error));
    }
    Ok(())
}
