//! FP16 prefix restore against cold chunked prefill on the unchanged oracle cohort.
#[path = "prefix_quality_kv8_device_tests.rs"]
mod kv8;

use super::super::{
    common_tests::{dims, inventory},
    prefill::ChunkViews,
    prefill_plan::ChunkSlot,
    state::export::f32_from_le,
};
use super::*;
use crate::family::gguf_model::open_qwen3_6_35b_a3b;
use crate::family::project::cuda::{CudaProjector, LoadOptions, Reserve};
use sha2::{Digest, Sha256};

const CONTEXT: usize = 2048;
const FIRST: usize = 1024;
const COUNT: usize = 2046;
const VOCAB: usize = 248_077;
const HEAD: usize = 248_320;
const IDS_SHA256: &str = "16d4b7e0fe2611467ebf10fb4f56f6074b47b8717a012e3488ba8cb34300c528";

fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("FP16 prefix cohort: {message}"))
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Tokens {
    bos: bool,
    ids: Vec<u32>,
}

fn cohort_ids(bytes: &[u8]) -> Result<Vec<u32>, EngError> {
    if format!("{:x}", Sha256::digest(bytes)) != IDS_SHA256 {
        return Err(bad("supplied tokens.json hash differs"));
    }
    let input: Tokens = serde_json::from_slice(bytes).map_err(|_| bad("invalid token JSON"))?;
    if input.bos || input.ids.len() != 4380 || input.ids.iter().any(|&id| id >= VOCAB as u32) {
        return Err(bad("BOS, token count, or sampler domain differs"));
    }
    Ok(input.ids)
}

fn positions(p0: usize, rows: usize) -> Result<std::ops::Range<usize>, EngError> {
    let end = p0
        .checked_add(rows)
        .filter(|&end| rows > 0 && end <= CONTEXT)
        .ok_or_else(|| bad("chunk outside the oracle context"))?
        .min(CONTEXT - 1);
    Ok(p0.max(FIRST).min(end)..end)
}

struct Row {
    logits: Vec<f32>,
    log_z: f64,
    nll: f64,
    top: usize,
}

fn row(logits: Vec<f32>, width: usize, teacher: usize) -> Result<Row, EngError> {
    if width < 2
        || width > logits.len()
        || teacher >= width
        || logits.iter().any(|v| !v.is_finite())
    {
        return Err(bad("invalid head, teacher, or sampler domain"));
    }
    let mut top = 0;
    for id in 1..width {
        if logits[id] > logits[top] {
            top = id;
        }
    }
    let max = f64::from(logits[top]);
    let log_z = max
        + logits[..width]
            .iter()
            .map(|&v| (f64::from(v) - max).exp())
            .sum::<f64>()
            .ln();
    let nll = log_z - f64::from(logits[teacher]);
    Ok(Row {
        logits,
        log_z,
        nll,
        top,
    })
}

struct Pair {
    reference_nll: f64,
    candidate_nll: f64,
    same_top: bool,
    kl: f64,
}

fn compare(reference: &Row, candidate: &Row, width: usize) -> Result<Pair, EngError> {
    if width < 2 || width > reference.logits.len() || width > candidate.logits.len() {
        return Err(bad("KL domain does not fit the heads"));
    }
    // Candidate-to-reference, matching the oracle's direction.
    let kl = reference.logits[..width]
        .iter()
        .zip(&candidate.logits[..width])
        .map(|(&r, &c)| {
            let log_p = f64::from(c) - candidate.log_z;
            let log_q = f64::from(r) - reference.log_z;
            log_p.exp() * (log_p - log_q)
        })
        .sum();
    Ok(Pair {
        reference_nll: reference.nll,
        candidate_nll: candidate.nll,
        same_top: reference.top == candidate.top,
        kl,
    })
}

fn read_range(step: &QwenResidentStep<'_>, range: DeviceRange<'_>) -> Result<Vec<u8>, EngError> {
    let mut bytes = vec![0u8; range.bytes()];
    range
        .buffer()
        .read_at(&step.stream, range.offset(), &mut bytes)?;
    Ok(bytes)
}

fn chunk_rows(
    step: &mut QwenResidentStep<'_>,
    sequence: &[u32],
    p0: usize,
    count: usize,
) -> Result<Vec<Row>, EngError> {
    let scored = positions(p0, count)?;
    let input = sequence
        .get(p0..p0 + count)
        .ok_or_else(|| bad("incomplete input context"))?;
    step.run_chunk(p0, input)?;
    let arena = step
        .prefill
        .as_ref()
        .ok_or_else(|| bad("no admitted arena"))?;
    let views = ChunkViews::new(arena.list(), arena.arena())?;
    let hidden = views.get(ChunkSlot::Hidden)?;
    let scratch = step.view(final_resid(step.res.table.len()))?;
    let row_bytes = step
        .res
        .geometry
        .hidden
        .checked_mul(F32)
        .ok_or_else(|| bad("hidden width overflows"))?;
    if scratch.bytes() != row_bytes {
        return Err(bad("tail scratch does not hold one hidden row"));
    }
    let mut rows = Vec::new();
    for position in scored {
        let offset = hidden
            .offset()
            .checked_add((position - p0) * row_bytes)
            .ok_or_else(|| bad("hidden offset overflows"))?;
        let bytes = read_range(step, hidden.buffer().range(offset, row_bytes)?)?;
        scratch
            .buffer()
            .write_at(&step.stream, scratch.offset(), &bytes)?;
        step.final_norm()?;
        step.lm_head()?;
        let bytes = read_range(step, step.view(Slot::Logits)?)?;
        if bytes.len() != HEAD * F32 {
            return Err(bad("logits do not cover the actual model head"));
        }
        rows.push(row(
            f32_from_le(&bytes),
            VOCAB,
            sequence[position + 1] as usize,
        )?);
    }
    Ok(rows)
}

struct Observed {
    restored: Vec<(usize, Vec<Pair>)>,
    per_token: Vec<Pair>,
}

fn device_phase(ids: &[u32]) -> Result<Observed, EngError> {
    // Resolve fixture helpers before acquiring CUDA owners: they contain expects.
    let dims = dims();
    let inv = inventory();
    let opts = StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: CONTEXT,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    };
    let rules = QwenRules::new(vec![248_046, 248_044], VOCAB, HEAD)?;
    let dir = std::env::var("Q36_CUDA_MODEL_DIR")
        .unwrap_or_else(|_| "/media/dan/nvme/models/qwen3.6-35b-a3b".to_owned());
    let model = open_qwen3_6_35b_a3b(
        &std::path::Path::new(&dir).join("Qwen3.6-35B-A3B-Q4_K_M.gguf"),
        None,
    )?;
    let projector = CudaProjector::new()?;
    let weights = projector
        .load_model(
            &model,
            LoadOptions {
                reserve: Reserve::default(),
                drop_page_cache: None,
                layout: WeightLayout::Native,
            },
        )?
        .device;
    let budget = projector
        .memory_info()?
        .free
        .checked_sub(1usize << 30)
        .ok_or_else(|| bad("no 1 GiB headroom after loading weights"))?;
    let ctx = projector.context();
    let res = StepResources::new(ctx, dims, &inv, &opts)?;
    let mut step = QwenResidentStep::new(ctx, &res, &weights, &opts, rules)?;
    step.set_capture_config(CaptureConfig {
        capture: false,
        section_marks: false,
    })?;
    step.set_m_sess_cap(CONTEXT as u32);
    step.admit_prefill(
        PrefillBudget {
            device_free: budget,
            extras: Vec::new(),
        },
        false,
        false,
        false,
    )?;
    if step.chunk_ready() != (ChunkVerdict::Ready { m_sess: CONTEXT }) {
        return Err(bad("real 2048-row FP16 dispatch was not admitted"));
    }
    let sequences = ids
        .get(..2 * CONTEXT)
        .ok_or_else(|| bad("two contexts are required"))?
        .as_chunks::<CONTEXT>()
        .0;
    let mut reference = Vec::new();
    for sequence in sequences {
        step.reset_state()?;
        reference.push(chunk_rows(&mut step, sequence, 0, CONTEXT)?);
    }
    let mut restored = Vec::new();
    for cut in [1024, 1536] {
        let mut pairs = Vec::new();
        for (index, sequence) in sequences.iter().enumerate() {
            step.reset_state()?;
            let mut candidate = chunk_rows(&mut step, sequence, 0, cut)?;
            let saved = step.export_state()?;
            // Overwrite the saved range with a different context, so restore cannot be a no-op.
            let foreign = &sequences[1 - index][..cut];
            if foreign == &sequence[..cut] {
                return Err(bad("foreign overwrite is not distinct"));
            }
            step.reset_state()?;
            step.run_chunk(0, foreign)?;
            step.import_state(&saved)?;
            if step.tokens() != cut {
                return Err(bad("restore did not recover the saved position"));
            }
            candidate.extend(chunk_rows(&mut step, sequence, cut, CONTEXT - cut)?);
            if candidate.len() != reference[index].len() || step.tokens() != CONTEXT {
                return Err(bad("restored context or scored positions are incomplete"));
            }
            for (r, c) in reference[index].iter().zip(&candidate) {
                pairs.push(compare(r, c, VOCAB)?);
            }
        }
        restored.push((cut, pairs));
    }
    let mut per_token = Vec::new();
    for (index, sequence) in sequences.iter().enumerate() {
        step.reset_state()?;
        for (position, &id) in sequence.iter().enumerate() {
            step.run_token(id, true)?;
            if (FIRST..CONTEXT - 1).contains(&position) {
                let candidate = row(
                    f32_from_le(step.row_bytes()),
                    VOCAB,
                    sequence[position + 1] as usize,
                )?;
                per_token.push(compare(
                    &reference[index][position - FIRST],
                    &candidate,
                    VOCAB,
                )?);
            }
        }
    }
    step.stream
        .synchronize()
        .inspect_err(|_| step.state.poison())?;
    Ok(Observed {
        restored,
        per_token,
    })
}

fn report(label: &str, pairs: &[Pair]) -> (f64, f64, f64) {
    assert_eq!(pairs.len(), COUNT);
    let mut signed = 0.0;
    let mut kl = 0.0;
    let mut same = 0usize;
    for (index, pair) in pairs.iter().enumerate() {
        signed += pair.candidate_nll - pair.reference_nll;
        kl += pair.kl;
        same += usize::from(pair.same_top);
        println!(
            "PREFIX_POSITION path={label} chunk={} input={} reference_nll={:.17} candidate_nll={:.17} kl={:.17} top1_same={}",
            index / 1023,
            FIRST + index % 1023,
            pair.reference_nll,
            pair.candidate_nll,
            pair.kl,
            pair.same_top
        );
    }
    let mean = signed / COUNT as f64;
    let mean_kl = kl / COUNT as f64;
    let agreement = same as f64 / COUNT as f64;
    println!(
        "PREFIX_GATE path={label} positions={COUNT} signed_dnll={mean:.12} mean_kl={mean_kl:.12} top1={agreement:.12}"
    );
    (mean, mean_kl, agreement)
}

#[test]
fn score_positions_cover_each_restore_cut() {
    for cut in [1024, 1536] {
        let mut selected: Vec<_> = positions(0, cut).unwrap().collect();
        selected.extend(positions(cut, CONTEXT - cut).unwrap());
        assert_eq!(selected, (FIRST..CONTEXT - 1).collect::<Vec<_>>());
        assert_eq!(selected.len() * 2, COUNT);
    }
    assert!(positions(usize::MAX, 1).is_err());
    assert!(positions(0, 0).is_err());
    assert!(cohort_ids(br#"{"bos":false,"ids":[]}"#).is_err());
}

#[test]
fn scoring_uses_signed_nll_and_candidate_to_reference_kl() {
    let uniform = row(vec![0.0, 0.0], 2, 0).unwrap();
    let skewed = row(vec![0.8f32.ln(), 0.2f32.ln()], 2, 0).unwrap();
    let same = compare(&uniform, &uniform, 2).unwrap();
    assert!(same.kl.abs() < 1e-12);
    assert!(same.same_top);
    let pair = compare(&skewed, &uniform, 2).unwrap();
    assert!((pair.kl - 0.22314355131420976).abs() < 1e-7);
    assert!(pair.candidate_nll - pair.reference_nll > 0.0);
    assert!(compare(&uniform, &skewed, 2).unwrap().candidate_nll - uniform.nll < 0.0);
    assert_eq!(row(vec![0.0, 0.0, 1000.0], 2, 0).unwrap().top, 0);
    assert!(row(vec![f32::NAN, 0.0], 2, 0).is_err());
    assert!(row(vec![0.0, 0.0], 2, 2).is_err());
    assert!(compare(&uniform, &uniform, 3).is_err());
}

#[test]
#[ignore = "full native model and immutable oracle cohort; CPU/GPU leases required"]
fn fp16_restore_matches_cold_chunked_on_oracle_cohort() {
    let dir = std::env::var("Q36_CUDA_COHORT_DIR").unwrap_or_else(|_| {
        "/media/dan/nvme/fifi/scrch/bba24a15/llamacpp-oracle/cohort".to_owned()
    });
    let bytes =
        std::fs::read(std::path::Path::new(&dir).join("tokens.json")).expect("supplied cohort");
    let ids = cohort_ids(&bytes).expect("unchanged supplied cohort");
    let observed = device_phase(&ids).expect("real FP16 cold/restore comparison");
    // No CUDA owners survive into the host assertions.
    report("per-token", &observed.per_token);
    for (cut, pairs) in observed.restored {
        let (signed, kl, agreement) = report(&format!("restore-{cut}"), &pairs);
        assert!(signed <= 0.0117, "signed restore-minus-cold NLL {signed}");
        assert!(kl <= 0.004, "mean candidate-to-cold KL {kl}");
        assert!(agreement >= 0.97, "top-1 agreement {agreement}");
    }
    println!("FP16_PREFIX_COHORT_PASS");
}
