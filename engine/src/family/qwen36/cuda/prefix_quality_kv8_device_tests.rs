//! The three-flag KV8 route, including a valid device tip and its overwritten host fallback.
use super::*;
use crate::family::qwen36::cuda::{
    prefill_plan::{Extra, PrefillStep, chunk_list},
    state::pinned::PinnedSnapshot,
};

const ROWS: usize = 512;

fn chunked(
    step: &mut QwenResidentStep<'_>,
    sequence: &[u32],
    range: std::ops::Range<usize>,
) -> Result<Vec<Row>, EngError> {
    let mut rows = Vec::new();
    for piece in chunk_list(range.start, range.len(), ROWS)? {
        let PrefillStep::Chunk { p0, rows: count } = piece else {
            return Err(bad("KV8 cohort unexpectedly planned a one-row feed"));
        };
        rows.extend(chunk_rows(step, sequence, p0, count)?);
    }
    Ok(rows)
}

fn snapshot(step: &mut QwenResidentStep<'_>) -> Result<PinnedSnapshot, EngError> {
    let pool = step
        .prefix_pins
        .as_mut()
        .ok_or_else(|| bad("KV8 pinned pool is absent"))?;
    let mut pending = pool.begin(step.context, &step.stream, &step.state, None)?;
    let tip = step
        .device_tip
        .as_mut()
        .ok_or_else(|| bad("KV8 device tip is absent"))?;
    pending.attach_device_tip(tip.capture(step.context, &step.stream, &mut step.state)?)?;
    pool.finish(step.context, &step.stream, &step.state, pending)
}

struct Case {
    cut: usize,
    foreign: bool,
    pairs: Vec<Pair>,
}

struct Kv8Observed {
    cases: Vec<Case>,
    per_token: Vec<Pair>,
}

fn device_phase(ids: &[u32]) -> Result<Kv8Observed, EngError> {
    for name in [
        "Q36_CUDA_CHUNK_PREFILL",
        "Q36_CUDA_PINNED_PREFIX",
        "Q36_CUDA_STATE_ONLY_PREFIX",
    ] {
        if std::env::var(name).as_deref() != Ok("1") {
            return Err(bad("KV8 qualification requires all three requested flags"));
        }
    }
    let dims = dims();
    let inv = inventory();
    let opts = StepOptions {
        kv_mode: KvMode::Int8G32,
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
        .ok_or_else(|| bad("KV8 has no 1 GiB headroom after weights"))?;
    let ctx = projector.context();
    let res = StepResources::new(ctx, dims, &inv, &opts)?;
    let mut step = QwenResidentStep::new(ctx, &res, &weights, &opts, rules)?;
    step.set_capture_config(CaptureConfig {
        capture: false,
        section_marks: false,
    })?;
    step.set_kv8_chunked_prefill(true)?;
    step.set_m_sess_cap(ROWS as u32);
    step.enable_pinned_prefix()?;
    let extra = step.enable_device_tip()?;
    if extra == 0 {
        return Err(bad("KV8 device tip allocation fell back"));
    }
    step.admit_prefill(
        PrefillBudget {
            device_free: budget,
            extras: vec![Extra {
                name: "device-prefix-delta",
                fixed: extra,
                bytes_per_token: 0,
            }],
        },
        false,
        false,
        false,
    )?;
    if step.chunk_ready() != (ChunkVerdict::Ready { m_sess: ROWS }) {
        return Err(bad("real 512-row KV8 dispatch was not admitted"));
    }
    let sequences = ids
        .get(..2 * CONTEXT)
        .ok_or_else(|| bad("two KV8 contexts are required"))?
        .as_chunks::<CONTEXT>()
        .0;
    let mut reference = Vec::new();
    for sequence in sequences {
        step.reset_state()?;
        reference.push(chunked(&mut step, sequence, 0..CONTEXT)?);
    }
    let mut cases = Vec::new();
    // 1280 also splits an admitted 512-row chunk, unlike the two aligned control cuts.
    for cut in [1024, 1280, 1536] {
        for foreign in [false, true] {
            let mut pairs = Vec::new();
            for (index, sequence) in sequences.iter().enumerate() {
                step.reset_state()?;
                let mut candidate = chunked(&mut step, sequence, 0..cut)?;
                let saved = snapshot(&mut step)?;
                step.reset_state()?;
                if foreign {
                    let overwrite = &sequences[1 - index][..cut];
                    if overwrite == &sequence[..cut] {
                        return Err(bad("KV8 foreign overwrite is not distinct"));
                    }
                    step.feed_rows_chunked(overwrite, ROWS)?;
                }
                let pool = step
                    .prefix_pins
                    .as_mut()
                    .ok_or_else(|| bad("KV8 lost its pool"))?;
                let device = pool.restore(
                    step.context,
                    &step.stream,
                    &mut step.state,
                    &saved,
                    step.device_tip.as_ref(),
                )?;
                if device == foreign || step.tokens() != cut {
                    return Err(bad("KV8 restore source or recovered position differs"));
                }
                candidate.extend(chunked(&mut step, sequence, cut..CONTEXT)?);
                if candidate.len() != reference[index].len() || step.tokens() != CONTEXT {
                    return Err(bad("KV8 restored scores or state are incomplete"));
                }
                for (r, c) in reference[index].iter().zip(&candidate) {
                    pairs.push(compare(r, c, VOCAB)?);
                }
            }
            cases.push(Case {
                cut,
                foreign,
                pairs,
            });
        }
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
    Ok(Kv8Observed { cases, per_token })
}

#[test]
fn kv8_restore_partitions_preserve_all_scored_positions() {
    for cut in [1024, 1280, 1536] {
        let mut selected = Vec::new();
        for range in [0..cut, cut..CONTEXT] {
            for piece in chunk_list(range.start, range.len(), ROWS).unwrap() {
                let PrefillStep::Chunk { p0, rows } = piece else {
                    panic!("one-row piece")
                };
                selected.extend(positions(p0, rows).unwrap());
            }
        }
        assert_eq!(selected, (FIRST..CONTEXT - 1).collect::<Vec<_>>());
        assert_eq!(selected.len() * 2, COUNT);
    }
}

#[test]
#[ignore = "full native KV8 model/cohort with three flags; CPU/GPU leases required"]
fn q8_state_only_restore_matches_cold_chunked_on_oracle_cohort() {
    let dir = std::env::var("Q36_CUDA_COHORT_DIR").unwrap_or_else(|_| {
        "/media/dan/nvme/fifi/scrch/bba24a15/llamacpp-oracle/cohort".to_owned()
    });
    let bytes =
        std::fs::read(std::path::Path::new(&dir).join("tokens.json")).expect("supplied cohort");
    let ids = cohort_ids(&bytes).expect("unchanged supplied cohort");
    let observed = device_phase(&ids).expect("real three-flag KV8 cold/restore comparison");
    report("kv8-per-token", &observed.per_token);
    for case in observed.cases {
        let label = format!(
            "kv8-{}-{}",
            if case.foreign { "host" } else { "device" },
            case.cut
        );
        let (signed, kl, agreement) = report(&label, &case.pairs);
        assert!(
            signed <= 0.0117,
            "KV8 signed restore-minus-cold NLL {signed}"
        );
        assert!(kl <= 0.004, "KV8 mean candidate-to-cold KL {kl}");
        assert!(agreement >= 0.97, "KV8 top-1 agreement {agreement}");
    }
    println!("KV8_PREFIX_COHORT_PASS");
}
