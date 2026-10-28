//! Loader-to-stack wiring over the real pinned checkpoint: the first-generation smoke.

use std::{path::Path, time::Instant};

use inference_engine::{EngError, family::nemotron35::runner, tokenizer::nemotron};

const PROMPT: &str = "The quick brown fox";

#[test]
fn generates_from_the_pinned_real_model_when_configured() {
    let Some(dir) = std::env::var_os("NEM35_MODEL_DIR") else {
        println!("nemotron35_runner: skipping, NEM35_MODEL_DIR is not set");
        return;
    };
    if !cfg!(target_os = "linux") {
        println!(
            "nemotron35_runner: skipping pinned generation; the availability probe requires Linux, and callers with explicit budgets use generate_with_admission"
        );
        return;
    }
    let dir = Path::new(&dir);
    let started = Instant::now();
    let bundle = nemotron::load(dir).expect("load pinned tokenizer bundle");
    let tokenizer_load_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let prompt = bundle.tokenizer().encode(PROMPT).expect("encode prompt");
    let prompt_encode_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let generated = runner::generate(dir, &prompt, 8, 32).expect("generate");
    let generate_wall_seconds = started.elapsed().as_secs_f64();

    assert!(
        generated.produced.iter().all(|&id| id < 131_072),
        "produced ids out of range: {:?}",
        generated.produced
    );
    assert!(
        generated.produced.windows(2).any(|pair| pair[0] != pair[1]),
        "all produced ids are identical: {:?}",
        generated.produced
    );
    let text = bundle
        .tokenizer()
        .decode(&generated.produced)
        .expect("decode produced ids");

    println!("PROMPT: {PROMPT:?}");
    println!("IDS: {:?}", generated.produced);
    println!("TEXT: {text:?}");
    println!(
        "elapsed_seconds: tokenizer_load={tokenizer_load_seconds:.3} prompt_encode={prompt_encode_seconds:.3} generate_wall={generate_wall_seconds:.3} generate_stack_reported={:.3}",
        generated.elapsed.as_secs_f64()
    );
}

#[test]
fn rejects_a_pinned_model_before_opening_the_loader_when_budget_is_too_small() {
    let Some(dir) = std::env::var_os("NEM35_MODEL_DIR") else {
        println!("nemotron35_runner: skipping admission, NEM35_MODEL_DIR is not set");
        return;
    };
    let error =
        runner::generate_with_admission(Path::new(&dir), &[], 1, 1, 20 * 1024 * 1024 * 1024)
            .expect_err("the pinned model cannot fit while preserving the 20 GiB floor");
    let EngError::MemoryPlan(message) = error else {
        panic!("expected a memory-plan rejection, got {error:?}");
    };
    assert!(message.contains("dense payload"), "{message}");
}
