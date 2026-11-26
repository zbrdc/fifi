//! The KV-cache-form quality leg: teacher-forced scoring of one cache form (fp16 or int8) over
//! long windows of a fixed token stream, on the GGUF weights and the chunked prefill path the
//! default start serves.
//!
//! One form per process, driven by the environment (the served API cannot return per-position
//! log-probabilities). Every window prefills the tokens before its scored region through the
//! chunked path, then feeds each true token one at a time and reads the full logits row of the
//! next position. The rows are narrowed to the sampler vocabulary and scored into the per-window
//! sidecars of `family::teacher_forced`, which also reads the stream, the windows and the
//! reference run from the environment.
//!
//! The pairing, the bootstrap and the verdict run on the host from the sidecars of two runs.

use super::{
    common_tests::{dims, inventory},
    dims::{KvMode, StepOptions},
    m3_run::{CHUNK_ROWS, SAMPLER_VOCAB, err, rules, stamp},
    prefill_driver::{FIXED_ADMISSION_MARGIN, PrefillBudget},
    resources::StepResources,
    step::{ChunkVerdict, QwenResidentStep},
};
use crate::{
    EngError,
    family::{
        gguf_model::open_qwen3_6_35b_a3b,
        project::cuda::{CudaProjector, LoadOptions, Reserve, WeightLayout},
        teacher_forced::{Run, Window},
    },
};
use std::path::PathBuf;

fn env(name: &str) -> Result<String, EngError> {
    std::env::var(name).map_err(|_| err(&format!("{name} is not set")))
}

/// One window on a fresh step: the prefill, then one forced step per scored position into the
/// window's sidecar.
fn run_window(
    step: &mut QwenResidentStep<'_>,
    kv: KvMode,
    free_after_load: usize,
    run: &Run,
    (w, (lo, score_lo, hi)): (usize, Window),
) -> Result<(), EngError> {
    let ids = &run.ids;
    let t0 = std::time::Instant::now();
    println!("kvq window {w} start: prefill {} ids", score_lo - lo);
    if kv == KvMode::Int8G32 {
        step.set_kv8_chunked_prefill(true)?;
    }
    step.set_m_sess_cap(CHUNK_ROWS);
    let verdict = step.admit_prefill(
        PrefillBudget {
            device_free: free_after_load.saturating_sub(FIXED_ADMISSION_MARGIN),
            extras: Vec::new(),
        },
        true,
        true,
        false,
    )?;
    if !matches!(verdict, ChunkVerdict::Ready { .. }) {
        return Err(err(&format!(
            "window {w}: the chunked arena verdict {verdict:?} is not Ready"
        )));
    }
    for piece in ids[lo..score_lo - 1].chunks(CHUNK_ROWS as usize) {
        step.feed_rows_chunked(piece, CHUNK_ROWS as usize)?;
    }
    println!("kvq window {w} prefill done [{}]", stamp(&t0));
    let mut sidecar = run.start(w)?;
    for p in score_lo..hi {
        step.run_token(ids[p - 1], true)?;
        let row = step
            .row_bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .take(SAMPLER_VOCAB)
            .map(|c| f32::from_le_bytes(*c))
            .collect::<Vec<f32>>();
        sidecar.push(&row, ids[p])?;
    }
    run.finish(sidecar)
}

/// Scores the windows `KVQ_WINDOWS` (`lo:hi`) with the cache form `KVQ_KV` (`f16` or `q8`).
#[test]
#[ignore = "the 3090 under fifi-gpu-run: KVQ_KV, KVQ_GGUF_DIR, KVQ_IDS, KVQ_IDS_SHA, KVQ_LAYOUT, KVQ_WINDOWS, KVQ_OUT, optional KVQ_REF"]
fn kv_quality_leg() {
    let kv = match env("KVQ_KV").expect("KVQ_KV").as_str() {
        "f16" => KvMode::Fp16,
        "q8" => KvMode::Int8G32,
        other => panic!("KVQ_KV is f16|q8, not {other}"),
    };
    let run = Run::from_env().expect("run");
    let projector = CudaProjector::new().expect("projector");
    let model = open_qwen3_6_35b_a3b(
        &PathBuf::from(env("KVQ_GGUF_DIR").expect("KVQ_GGUF_DIR"))
            .join("Qwen3.6-35B-A3B-Q4_K_M.gguf"),
        None,
    )
    .expect("model");
    let weights = projector
        .load_model(
            &model,
            LoadOptions {
                reserve: Reserve {
                    kv_cache: 0,
                    work: 0,
                    headroom: 0,
                },
                drop_page_cache: None,
                layout: WeightLayout::Native,
            },
        )
        .expect("load")
        .device;
    let layout_kind = weights.layout().expect("layout kind");
    let free_after_load = projector.memory_info().expect("memory").free;
    println!("kvq kv={:?} free after load {free_after_load}", kv);
    let options = StepOptions {
        kv_mode: kv,
        context_tokens: run.context + CHUNK_ROWS as usize,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: layout_kind,
    };
    let res =
        StepResources::new(projector.context(), dims(), &inventory(), &options).expect("resources");
    for window in run.pending() {
        let mut step =
            QwenResidentStep::new(projector.context(), &res, &weights, &options, rules())
                .expect("step");
        run_window(&mut step, kv, free_after_load, &run, window)
            .unwrap_or_else(|e| panic!("window {}: {e:?}", window.0));
    }
}
