//! Format-token probe, committed as an ignored test. The probability of "\n" right after the
//! tool-call opener on the 922-token pi prompt, whole model, eager, in the layout `FMT_LAYOUT`
//! names (`native` or `tiled`, the latter by default). `FMT_IDS` is a one-line space-separated id
//! file (the prompt ids, then `</think>`, `\n\n`, `<tool_call>`); `FMT_NL` is the id of "\n" (198);
//! `FMT_OUT` a JSON path. The probe reports the softmax over the sampler vocabulary at the last 4
//! rows and gates nothing. It needs the 3090, `INFERENCE_CUDA_MODEL_GGUF` and the three `FMT_*`
//! variables, so it is `#[ignore]`d and run by name while it holds the 3090 exclusively:
//! `<exe> --ignored --exact family::qwen36::cuda::r5_fmt::r5_format_token --test-threads=1`.
//! The optional `FMT_TIERED` is a JSON object in the shape of the gate's `tiered` member
//! (`{"slots": 40, "threads": 6, "preload": true}`) and runs the tiered expert mode: the layout
//! must be `native`, the routed expert stacks stay in the file and the banks are attached after
//! the step, as in the gate, and the output JSON carries the member as `tiered`. Unset, nothing
//! changes: the plain probe and its output are byte for byte as before.
//! It is a probe, run alone in its own process: its device phase uses `expect`, so a failure
//! latches only that process.

use std::{path::PathBuf, time::Instant};

use super::{
    super::host::HostShape,
    dims::{KvMode, StepDims, StepOptions, TieredPlan},
    gate::tiered_member,
    resources::StepResources,
    rules::QwenRules,
    step::QwenResidentStep,
    tiered_load::{attach_tiered_parts, names_without_experts},
};
use crate::{
    family::{
        QWEN3_6_35B_A3B,
        gguf_model::{Inventory, open_qwen3_6_35b_a3b},
        project::cuda::{CudaProjector, LoadOptions, Reserve, WeightLayout},
    },
    kernels::cuda::device_count,
};

const SAMPLER_VOCAB: usize = 248_077;
const HEAD_ROWS: usize = 248_320;
const EOS: [u32; 2] = [248_046, 248_044];
const CONTEXT_TOKENS: usize = 1024;
const TAIL_ROWS: usize = 4;

fn required(var: &str) -> String {
    std::env::var(var)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| panic!("set {var}"))
}

/// The tiered plan of `FMT_TIERED` (the gate's member text, checked by the gate's own reader), or `None` when unset.
fn tiered_from_env() -> Option<(TieredPlan, String)> {
    let text = std::env::var("FMT_TIERED").ok().filter(|v| !v.is_empty())?;
    let member: serde_json::Value = serde_json::from_str(&text).expect("FMT_TIERED is JSON");
    let holder = serde_json::json!({ "tiered": member });
    let plan = tiered_member(&holder)
        .expect("FMT_TIERED is a valid tiered member")
        .expect("FMT_TIERED names the member");
    Some((plan, holder["tiered"].to_string()))
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Softmax stats of one logits row over the sampler vocabulary: (log-sum-exp, top-5 ids).
fn row_stats(logits: &[f32], watch: usize) -> String {
    let row = &logits[..SAMPLER_VOCAB];
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|v| f64::from(v - max).exp()).sum();
    let prob = |id: usize| f64::from(row[id] - max).exp() / sum;
    let mut order: Vec<usize> = (0..SAMPLER_VOCAB).collect();
    order.sort_unstable_by(|a, b| row[*b].partial_cmp(&row[*a]).expect("finite logits"));
    let rank = order
        .iter()
        .position(|id| *id == watch)
        .expect("watch id in range");
    let top: Vec<String> = order
        .iter()
        .take(5)
        .map(|id| {
            format!(
                "{{\"id\":{id},\"logit\":{:.5},\"p\":{:.6e}}}",
                row[*id],
                prob(*id)
            )
        })
        .collect();
    format!(
        "{{\"p_watch\":{:.9e},\"logit_watch\":{:.5},\"rank_watch\":{rank},\"top5\":[{}]}}",
        prob(watch),
        row[watch],
        top.join(",")
    )
}

/// The `r5-fmt/1` output document. `member` is the tiered member text of a tiered run; a plain run passes `None` and the
/// document is byte for byte what the probe wrote before the member existed.
fn output_json(
    n_ids: usize,
    watch: usize,
    secs: f64,
    member: Option<&str>,
    rows: &[String],
) -> String {
    let member = member.map_or_else(String::new, |text| format!(",\"tiered\":{text}"));
    format!(
        "{{\"schema\":\"r5-fmt/1\",\"n_ids\":{n_ids},\"watch\":{watch},\"secs\":{secs:.2}{member},\"tail_rows\":[{}]}}\n",
        rows.join(",")
    )
}

#[test]
fn the_output_document_carries_the_tiered_member_only_when_given() {
    let rows = [String::from("{\"row\":1}"), String::from("{\"row\":2}")];
    assert_eq!(
        output_json(922, 198, 12.345, None, &rows),
        "{\"schema\":\"r5-fmt/1\",\"n_ids\":922,\"watch\":198,\"secs\":12.35,\"tail_rows\":[{\"row\":1},{\"row\":2}]}\n"
    );
    let tiered = output_json(922, 198, 12.345, Some("{\"slots\":40}"), &rows);
    assert!(
        tiered.contains("\"secs\":12.35,\"tiered\":{\"slots\":40},\"tail_rows\""),
        "{tiered}"
    );
    let parsed: serde_json::Value = serde_json::from_str(&tiered).expect("valid JSON");
    assert_eq!(parsed["tiered"]["slots"], 40);
}

#[test]
#[ignore = "device probe: needs the 3090, the real GGUF and FMT_IDS, FMT_NL, FMT_OUT; run by name"]
fn r5_format_token() {
    if !std::env::var("FMT_IDS").is_ok_and(|v| !v.is_empty()) {
        println!("SKIPPED r5_format_token: FMT_IDS is not set");
        return;
    }
    let n_dev = device_count().unwrap_or(0);
    println!("R5 fmt: CUDA devices visible: {n_dev}");
    assert!(n_dev > 0, "SKIPPED: no CUDA device");
    let ids: Vec<u32> = std::fs::read_to_string(PathBuf::from(required("FMT_IDS")))
        .expect("read FMT_IDS")
        .split_whitespace()
        .map(|t| t.parse().expect("id"))
        .collect();
    let watch: usize = required("FMT_NL").parse().expect("FMT_NL");
    let out_path = required("FMT_OUT");
    assert!(
        ids.len() < CONTEXT_TOKENS,
        "{} ids exceed the context",
        ids.len()
    );
    println!("R5 fmt: {} ids, watch token {watch}", ids.len());

    let path = required("INFERENCE_CUDA_MODEL_GGUF");
    let gguf = open_qwen3_6_35b_a3b(std::path::Path::new(&path), None).expect("open the gguf");
    let projector = CudaProjector::new().expect("projector");
    let native = std::env::var("FMT_LAYOUT").as_deref() == Ok("native");
    println!("R5 fmt: layout native = {native}");
    let load_options = LoadOptions {
        reserve: Reserve::default(),
        drop_page_cache: None,
        layout: if native {
            WeightLayout::Native
        } else {
            WeightLayout::Tiled
        },
    };
    let tiered = tiered_from_env();
    assert!(
        tiered.is_none() || native,
        "FMT_TIERED needs FMT_LAYOUT=native (the tiered mode refuses the tiled layout)"
    );
    let device = if tiered.is_some() {
        // The routed expert stacks stay in the file: the banks hold the hot ones.
        let names = gguf.plan().device_names();
        let names = names_without_experts(&names);
        projector
            .load(&gguf.weights, &names, load_options)
            .expect("load the model without the expert stacks")
    } else {
        projector
            .load_model(&gguf, load_options)
            .expect("load the whole model")
            .device
    };
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).expect("shape");
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    let dims = StepDims::new(&shape, &inventory).expect("dims");
    assert_eq!(dims.shape.vocab, HEAD_ROWS, "the padded head width");
    let options = StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: CONTEXT_TOKENS,
        delta_slots: 1,
        forked_experts: false,
        tiered: tiered.as_ref().map(|(plan, _)| *plan),
        layout: device.layout().expect("the loaded layout"),
    };
    let context = projector.context();
    let res = StepResources::new(context, dims, &inventory, &options).expect("resources");
    let rules = QwenRules::new(EOS.to_vec(), SAMPLER_VOCAB, HEAD_ROWS).expect("rules");
    let mut step = QwenResidentStep::new(context, &res, &device, &options, rules).expect("step");
    if let Some((plan, _)) = &tiered {
        let source = (gguf.weights.header(), std::path::Path::new(&path));
        attach_tiered_parts(&mut step, context, (source, &res.dims), plan).expect("attach tiered");
    }

    let began = Instant::now();
    let mut rows = Vec::new();
    for (row, token) in ids.iter().enumerate() {
        let pair = step.run_token(*token, true).expect("run_token");
        if row + TAIL_ROWS >= ids.len() {
            let stats = row_stats(&f32s(step.row_bytes()), watch);
            println!("R5 fmt: row {row} tok {token} pair {pair} {stats}");
            rows.push(format!(
                "{{\"row\":{row},\"tok\":{token},\"pair\":{pair},\"stats\":{stats}}}"
            ));
        }
        if row % 100 == 0 {
            println!(
                "R5 fmt: row {row}/{} ({:.1} s)",
                ids.len(),
                began.elapsed().as_secs_f64()
            );
        }
    }
    let member = tiered.as_ref().map(|(_, text)| text.as_str());
    let json = output_json(
        ids.len(),
        watch,
        began.elapsed().as_secs_f64(),
        member,
        &rows,
    );
    std::fs::write(&out_path, json).expect("write FMT_OUT");
    println!("R5 fmt: PASS (probe; numbers reported, never gated), wrote {out_path}");
}
