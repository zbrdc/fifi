//! The Inkling quality leg: teacher-forced scoring of a fixed token stream on one GGUF file,
//! through the CUDA forward the server runs. Each window's prompt goes in the served prompt
//! chunks (`FIFI_INKLING_CUDA_PREFILL_ROWS`, 512 when unset) and each scored position is a decode
//! token, so the process's switches select the arm. The sidecars are those of
//! `family::teacher_forced`.

use super::CpuForward;
use crate::family::{
    inkling2::{TextBackend, serve::prefill_rows},
    teacher_forced::{Run, score_windows},
};
use std::path::Path;

#[test]
#[ignore = "the 3090 under fifi-gpu-run: KVQ_MODEL (an Inkling GGUF), KVQ_IDS, KVQ_IDS_SHA, KVQ_LAYOUT, KVQ_WINDOWS, KVQ_OUT, optional KVQ_REF"]
fn inkling_quality_leg() {
    let run = Run::from_env().expect("run");
    let model = std::env::var_os("KVQ_MODEL").expect("KVQ_MODEL");
    let mut forward = CpuForward::open(Path::new(&model), run.context + 1).expect("model");
    forward.enable_cuda(0).expect("CUDA");
    assert!(forward.supports_grouped_prefill(), "grouped prompt prefill");
    let chunk = prefill_rows(TextBackend::Cuda).expect("prompt chunk");
    let vocab = forward.spec.unpadded_vocab_size as usize;
    println!("kvq Inkling prompt chunk {chunk} vocabulary {vocab}");
    score_windows(&mut forward, &run, chunk, vocab).expect("windows");
}
