// Runs the same start/worker/executor path as serving, with already-tokenized input.
use crate::EngError;
use crate::family::executor::GenerationState;
use crate::family::qwen36::settings::{Backend, Q36Env, Q36Options, Q36Settings};
use crate::sched::job::{Job, JobEvent, JobUsage};
use crate::sched::worker::{EngineHandle, EngineState};
use crate::tokenizer::{IncrementalDecoder, PromptMessage, Role, Thinking, TokenizerBundle};
use std::path::Path;
use std::time::{Duration, Instant};

fn ttft_error(message: impl Into<String>) -> EngError {
    EngError::BadRequest(message.into())
}

fn serving_fixture_options(context_limit: u32) -> Result<Q36Options, EngError> {
    serde_json::from_value(serde_json::json!({
        "context_limit": context_limit, "kv_dtype": "q8", "capture": false
    }))
    .map_err(|error| ttft_error(error.to_string()))
}

#[test]
fn serving_fixture_configs_parse_without_engine() {
    for context in [65_536, 131_072] {
        let options = serving_fixture_options(context).expect("dry parse actual fixture config");
        assert_eq!(options.context_limit, Some(context));
        assert_eq!(options.kv_dtype.as_deref(), Some("q8"));
        assert_eq!(options.capture, Some(false));
    }
}

fn wait_ttft_ready(handle: &EngineHandle) -> Result<(), EngError> {
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        match handle.state() {
            EngineState::Ready => return Ok(()),
            EngineState::Loading if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(20));
            }
            state => {
                return Err(ttft_error(format!(
                    "TTFT engine {state:?}: {:?}",
                    handle.failure()
                )));
            }
        }
    }
}

fn time_served_job(
    handle: &EngineHandle,
    tokenizer: &TokenizerBundle,
    ids: Vec<u32>,
    max_tokens: u32,
) -> Result<(Duration, JobUsage, String), EngError> {
    let generation = GenerationState::new(IncrementalDecoder::new(tokenizer.tokenizer()))
        .with_max_tokens(max_tokens);
    let (job, mut owner) = Job::new(ids, generation)?;
    let began = Instant::now();
    handle
        .try_submit(job)
        .map_err(|error| ttft_error(format!("TTFT submit: {error:?}")))?;
    let mut first = None;
    let mut reply = String::new();
    while let Some(event) = owner.events.blocking_recv() {
        match event {
            JobEvent::Batch(batch) => {
                first.get_or_insert_with(|| began.elapsed());
                reply.push_str(&batch.text);
            }
            JobEvent::Finished(usage) => {
                if usage.completion_tokens == 0 || usage.reused_tokens != 0 {
                    return Err(ttft_error("TTFT needs a generated token and a cold prompt"));
                }
                return Ok((
                    first.ok_or_else(|| ttft_error("TTFT missing token batch"))?,
                    usage,
                    reply,
                ));
            }
            JobEvent::Error(error) => return Err(error),
        }
    }
    Err(ttft_error("TTFT event channel closed without usage"))
}

fn run_serving_ttft() -> Result<(), EngError> {
    let model_dir =
        std::env::var("Q36_CUDA_MODEL_DIR").map_err(|error| ttft_error(error.to_string()))?;
    let input_dir =
        std::env::var("Q36_CUDA_BENCH_INPUT_DIR").map_err(|error| ttft_error(error.to_string()))?;
    let mut inputs = Vec::new();
    for count in [8192usize, 32768] {
        let path = Path::new(&input_dir).join(format!("{count}.ids.json"));
        let bytes = std::fs::read(&path)
            .map_err(|error| ttft_error(format!("{}: {error}", path.display())))?;
        let ids: Vec<u32> =
            serde_json::from_slice(&bytes).map_err(|error| ttft_error(error.to_string()))?;
        if ids.len() != count || ids.iter().any(|&id| id >= 248_077) {
            return Err(ttft_error(
                "TTFT input token count or vocabulary is invalid",
            ));
        }
        inputs.push(ids);
    }
    let options = serving_fixture_options(65_536)?;
    let dir = Path::new(&model_dir);
    let settings = Q36Settings::new(dir, dir, Some(&options), &Q36Env::default(), Backend::Cuda)?;
    let (engine, tokenizer) = super::start(&settings, None)?;
    let handle = engine.handle();
    let measured = (|| {
        wait_ttft_ready(&handle)?;
        for ids in inputs {
            for repeat in 1..=2 {
                let (elapsed, usage, _) = time_served_job(&handle, &tokenizer, ids.clone(), 1)?;
                println!(
                    "TTFT flag={} rows={} prompt={} repeat={} ttft_ms={:.3} prefill_tok_s_inclusive_first={:.3} generated={} reused={}",
                    std::env::var("Q36_CUDA_CHUNK_PREFILL").unwrap_or_else(|_| "1".into()),
                    std::env::var("Q36_CUDA_CHUNK_ROWS").unwrap_or_else(|_| "1024".into()),
                    usage.prompt_tokens,
                    repeat,
                    elapsed.as_secs_f64() * 1000.0,
                    f64::from(usage.prompt_tokens.saturating_sub(1)) / elapsed.as_secs_f64(),
                    usage.completion_tokens,
                    usage.reused_tokens
                );
            }
        }
        Ok::<(), EngError>(())
    })();
    let shutdown = engine.shutdown();
    measured?;
    if let Some(error) = shutdown.failure {
        return Err(ttft_error(error));
    }
    Ok(())
}

#[test]
#[ignore = "actual serving-worker TTFT; requires full model, tokenized 8K/32K inputs and measurement locks"]
fn serving_chunk_prefill_ttft() {
    run_serving_ttft().expect("actual serving-worker TTFT");
}

fn long_sanity_input(bundle: &TokenizerBundle) -> Result<Vec<u32>, EngError> {
    let readme = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../README.md"));
    let mut body = bundle.tokenizer().encode(&readme.repeat(200))?;
    if body.len() < 98_000 {
        return Err(ttft_error("long sanity document is too short"));
    }
    body.truncate(98_000);
    // Cut the body before chat rendering so the generation delimiters remain intact.
    let message = PromptMessage {
        role: Role::User,
        content: Some(format!(
            "Reference documentation:\n{}\n\nWrite at least 300 characters explaining this inference engine and its supported backends. Use complete sentences.",
            bundle.tokenizer().decode(&body)?
        )),
        reasoning_content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    };
    let ids = bundle.encode_prompt(&[message], &[], Thinking::Off)?;
    if !(96_000..100_000).contains(&ids.len()) {
        return Err(ttft_error(format!(
            "long sanity prompt has {} tokens",
            ids.len()
        )));
    }
    Ok(ids)
}

fn run_serving_long_sanity() -> Result<(), EngError> {
    let model_dir =
        std::env::var("Q36_CUDA_MODEL_DIR").map_err(|error| ttft_error(error.to_string()))?;
    let options = serving_fixture_options(131_072)?;
    let dir = Path::new(&model_dir);
    let settings = Q36Settings::new(dir, dir, Some(&options), &Q36Env::default(), Backend::Cuda)?;
    let (engine, tokenizer) = super::start(&settings, None)?;
    let handle = engine.handle();
    let measured = (|| {
        wait_ttft_ready(&handle)?;
        let ids = long_sanity_input(&tokenizer)?;
        let (elapsed, usage, reply) = time_served_job(&handle, &tokenizer, ids, 256)?;
        if usage.prompt_tokens <= 65_536 || reply.chars().count() < 200 {
            return Err(ttft_error(
                "long sanity needs above-cap prefill and 200 reply characters",
            ));
        }
        let first200: String = reply.chars().take(200).collect();
        println!(
            "LONG_SANITY prompt={} context=131072 ttft_ms={:.3} generated={} reused={} first200={first200:?}",
            usage.prompt_tokens,
            elapsed.as_secs_f64() * 1000.0,
            usage.completion_tokens,
            usage.reused_tokens
        );
        Ok::<(), EngError>(())
    })();
    let shutdown = engine.shutdown();
    measured?;
    if let Some(error) = shutdown.failure {
        return Err(ttft_error(error));
    }
    Ok(())
}

#[test]
#[ignore = "full model, above-64K fallback, 200 reply characters; measurement locks required"]
fn serving_kv8_long_context_sanity() {
    run_serving_long_sanity().expect("actual long-context serving sanity");
}
