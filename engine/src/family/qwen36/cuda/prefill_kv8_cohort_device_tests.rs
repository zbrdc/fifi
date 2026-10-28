// Same oracle IDs and scored positions; compare native all-layer paths, never a proxy.
const COHORT_TOKENS_SHA256: &str =
    "16d4b7e0fe2611467ebf10fb4f56f6074b47b8717a012e3488ba8cb34300c528";
const COHORT_CONTEXT: usize = 2048;
const COHORT_CHUNK: usize = 512;
const COHORT_FIRST: usize = 1024;
const COHORT_COUNT: usize = 2046;
const DEEP_PREFIX: usize = 65_536;
const DEEP_COUNT: usize = 1024;
const DEEP_CAPACITY: usize = 131_072;

#[derive(Debug, Clone, Copy)]
struct DeepCase {
    prefix: usize,
    capacity: usize,
    rows: usize,
    fixed: bool,
}

const DEEP_LEGACY: DeepCase = DeepCase {
    prefix: DEEP_PREFIX,
    capacity: DEEP_CAPACITY,
    rows: 1024,
    fixed: false,
};
const DEEP_FIXED: DeepCase = DeepCase {
    prefix: 196_608,
    capacity: 262_144,
    rows: 512,
    fixed: true,
};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CohortTokens {
    bos: bool,
    ids: Vec<u32>,
}

fn cohort_ids(bytes: &[u8]) -> Result<Vec<u32>, EngError> {
    if format!("{:x}", Sha256::digest(bytes)) != COHORT_TOKENS_SHA256 {
        return Err(bad("oracle cohort tokens.json hash differs"));
    }
    let input: CohortTokens =
        serde_json::from_slice(bytes).map_err(|_| bad("invalid oracle cohort tokens.json"))?;
    if input.bos
        || input.ids.len() != 4380
        || input.ids.iter().any(|&id| id >= SAMPLER_VOCAB as u32)
    {
        return Err(bad("oracle cohort BOS, count, or vocabulary differs"));
    }
    Ok(input.ids)
}

fn cohort_positions(p0: usize, rows: usize) -> Result<std::ops::Range<usize>, EngError> {
    let end = p0
        .checked_add(rows)
        .filter(|&end| rows != 0 && end <= COHORT_CONTEXT)
        .ok_or_else(|| bad("oracle cohort chunk is outside its context"))?
        .min(COHORT_CONTEXT - 1);
    Ok(p0.max(COHORT_FIRST).min(end)..end)
}

fn cohort_head_values(bytes: &[u8]) -> Result<Vec<f32>, EngError> {
    if bytes.len() != HEAD_ROWS * F32 {
        return Err(bad("oracle cohort logits do not cover the actual head"));
    }
    Ok(f32_from_le(bytes))
}

// Match the oracle direction: KL(chunked || per-token), on the production sampler domain.
fn cohort_kl(reference: &[f32], candidate: &[f32], width: usize) -> Result<f64, EngError> {
    let reference = reference
        .get(..width)
        .filter(|_| width >= 2)
        .ok_or_else(|| bad("cohort KL reference domain is invalid"))?;
    let candidate = candidate
        .get(..width)
        .ok_or_else(|| bad("cohort KL candidate domain is invalid"))?;
    if reference.iter().chain(candidate).any(|v| !v.is_finite()) {
        return Err(bad("cohort KL has non-finite logits"));
    }
    let normalizer = |row: &[f32]| {
        let max = row
            .iter()
            .map(|&v| f64::from(v))
            .fold(f64::NEG_INFINITY, f64::max);
        let log_sum = row
            .iter()
            .map(|&v| (f64::from(v) - max).exp())
            .sum::<f64>()
            .ln();
        (max, log_sum)
    };
    let (rmax, rsum) = normalizer(reference);
    let (cmax, csum) = normalizer(candidate);
    Ok(reference
        .iter()
        .zip(candidate)
        .map(|(&r, &c)| {
            let log_p = (f64::from(c) - cmax) - csum;
            let log_q = (f64::from(r) - rmax) - rsum;
            log_p.exp() * (log_p - log_q)
        })
        .sum())
}

fn cohort_row_score(
    step: &QwenResidentStep<'_>,
    row: usize,
    target: u32,
    reference: &CohortReference,
) -> Result<CohortCandidate, EngError> {
    let arena = step
        .prefill
        .as_ref()
        .ok_or_else(|| bad("no cohort arena"))?;
    let views = ChunkViews::new(arena.list(), arena.arena())?;
    let hidden = views.get(ChunkSlot::Hidden)?;
    let scratch = step.view(final_resid(step.res.table.len()))?;
    let row_bytes = step
        .res
        .geometry
        .hidden
        .checked_mul(F32)
        .ok_or_else(|| bad("cohort hidden width overflows"))?;
    if scratch.bytes() != row_bytes {
        return Err(bad("cohort tail scratch is not one hidden row"));
    }
    let relative = row
        .checked_mul(row_bytes)
        .ok_or_else(|| bad("cohort hidden row offset overflows"))?;
    let end = relative
        .checked_add(row_bytes)
        .filter(|&end| end <= hidden.bytes())
        .ok_or_else(|| bad("cohort hidden row exceeds its slot"))?;
    let offset = hidden
        .offset()
        .checked_add(relative)
        .ok_or_else(|| bad("cohort hidden slot offset overflows"))?;
    let bytes = read_range(&step.stream, hidden.buffer().range(offset, end - relative)?)?;
    scratch
        .buffer()
        .write_at(&step.stream, scratch.offset(), &bytes)?;
    step.final_norm()?;
    step.lm_head()?;
    let bytes = read_range(&step.stream, step.view(Slot::Logits)?)?;
    let logits = cohort_head_values(&bytes)?;
    let scored = score(&logits, SAMPLER_VOCAB, target as usize).map_err(bad)?;
    let kl = cohort_kl(&reference.logits, &logits, SAMPLER_VOCAB)?;
    Ok(CohortCandidate {
        score: scored,
        kl,
        logits,
    })
}

fn cohort_chunk_scores(
    step: &QwenResidentStep<'_>,
    sequence: &[u32],
    p0: usize,
    reference: &[CohortReference],
) -> Result<Vec<CohortCandidate>, EngError> {
    if sequence.len() != COHORT_CONTEXT {
        return Err(bad("cohort tail needs one complete oracle context"));
    }
    let arena = step
        .prefill
        .as_ref()
        .ok_or_else(|| bad("no cohort arena"))?;
    let mut scores = Vec::new();
    for position in cohort_positions(p0, arena.m_sess())? {
        let baseline = reference
            .get(position - COHORT_FIRST)
            .ok_or_else(|| bad("cohort KL has no corresponding reference row"))?;
        scores.push(cohort_row_score(
            step,
            position - p0,
            sequence[position + 1],
            baseline,
        )?);
    }
    Ok(scores)
}

struct CohortReference {
    score: Score,
    logits: Vec<f32>,
}

struct CohortCandidate {
    score: Score,
    kl: f64,
    logits: Vec<f32>,
}

struct CohortRows {
    reference: Vec<CohortReference>,
    candidate: Vec<CohortCandidate>,
}

fn cohort_device_phase(
    ids: &[u32],
    chunk_rows: usize,
    deep: Option<DeepCase>,
) -> Result<CohortRows, EngError> {
    if let Some(case) = deep {
        if ids.len() != case.prefix + DEEP_COUNT + 1 || chunk_rows != case.rows {
            return Err(bad(
                "deep cohort needs its prefix, 1024 inputs and configured dispatch",
            ));
        }
        if std::env::var("Q36_CUDA_KV8_STAGE_CONTEXT").as_deref() != Ok("1")
            || (case.fixed && std::env::var("Q36_CUDA_KV8_FIXED_ADMISSION").as_deref() != Ok("1"))
        {
            return Err(bad(
                "deep cohort requires its explicit staging/admission flags",
            ));
        }
    }
    let sequences = ids
        .get(..2 * COHORT_CONTEXT)
        .ok_or_else(|| bad("cohort needs two complete oracle contexts"))?;
    // Fixture helpers can panic; resolve them before acquiring any CUDA owner.
    let dims = step_dims();
    let inv = inventory();
    let mut opts = options();
    opts.context_tokens = deep.map_or(COHORT_CONTEXT, |case| case.capacity);
    let rules = rules()?;
    let dir = std::env::var("Q36_CUDA_MODEL_DIR")
        .unwrap_or_else(|_| "/media/dan/nvme/models/qwen3.6-35b-a3b".to_owned());
    let model = open_qwen3_6_35b_a3b(
        &std::path::Path::new(&dir).join("Qwen3.6-35B-A3B-Q4_K_M.gguf"),
        None,
    )?;
    let projector = CudaProjector::new()?;
    let fixed = deep.is_some_and(|case| case.fixed);
    let load = if fixed {
        super::super::start::load_options(
            super::super::start::kv_cache_bytes(opts.context_tokens, opts.kv_mode)?,
            true,
        )
    } else {
        LoadOptions {
            reserve: Reserve::default(),
            drop_page_cache: None,
            layout: WeightLayout::Native,
        }
    };
    let weights = projector.load_model(&model, load)?.device;
    let free = projector.memory_info()?.free;
    if fixed && free < super::super::pinned::MIN_FREE_AFTER_LOAD {
        return Err(EngError::MemoryPlan(
            "fixed cohort is below the loader free floor".into(),
        ));
    }
    let budget = free
        .checked_sub(super::super::start::prefill_headroom(fixed))
        .ok_or_else(|| bad("cohort has no admission headroom"))?;
    let context = projector.context();
    let res = StepResources::new(context, dims, &inv, &opts)?;
    if let Some(case) = deep {
        let mut step = QwenResidentStep::new(context, &res, &weights, &opts, rules)?;
        return cohort_deep_tail(&mut step, ids, budget, case, false);
    }
    let reference = {
        let mut step = QwenResidentStep::new(context, &res, &weights, &opts, rules.clone())?;
        step.set_capture_config(CaptureConfig {
            capture: false,
            section_marks: false,
        })?;
        let mut scores = Vec::with_capacity(COHORT_COUNT);
        for sequence in sequences.as_chunks::<COHORT_CONTEXT>().0 {
            step.reset_state()?;
            for (position, &id) in sequence.iter().enumerate() {
                step.run_token(id, true)?;
                if (COHORT_FIRST..COHORT_CONTEXT - 1).contains(&position) {
                    let logits = cohort_head_values(step.row_bytes())?;
                    let scored = score(&logits, SAMPLER_VOCAB, sequence[position + 1] as usize)
                        .map_err(bad)?;
                    scores.push(CohortReference {
                        score: scored,
                        logits,
                    });
                }
            }
            if boundary(&step)
                != (Boundary {
                    tokens: COHORT_CONTEXT,
                    health: Health::Valid,
                })
            {
                return Err(bad(
                    "cohort reference did not end with valid complete state",
                ));
            }
        }
        scores
    };
    let mut step = QwenResidentStep::new(context, &res, &weights, &opts, rules)?;
    step.set_capture_config(CaptureConfig {
        capture: false,
        section_marks: false,
    })?;
    step.set_kv8_chunked_prefill(true)?;
    step.set_m_sess_cap(chunk_rows as u32);
    step.admit_prefill(
        PrefillBudget {
            device_free: budget,
            extras: Vec::new(),
        },
        false,
        false,
        false,
    )?;
    if step.chunk_ready() != (ChunkVerdict::Ready { m_sess: chunk_rows }) {
        return Err(bad("cohort candidate did not admit the requested dispatch"));
    }
    let arena = step
        .prefill
        .as_ref()
        .ok_or_else(|| bad("cohort candidate has no arena"))?;
    if arena.kv8_staging()?.is_none() {
        return Err(bad("cohort candidate did not admit FP16 staging"));
    }
    let mut candidate = Vec::with_capacity(COHORT_COUNT);
    let per_context = COHORT_CONTEXT - COHORT_FIRST - 1;
    for (sequence_index, sequence) in sequences.as_chunks::<COHORT_CONTEXT>().0.iter().enumerate() {
        let start = sequence_index * per_context;
        let baseline = reference
            .get(start..start + per_context)
            .ok_or_else(|| bad("cohort KL reference context is incomplete"))?;
        step.reset_state()?;
        for chunk in sequence.chunks_exact(chunk_rows) {
            let p0 = step.tokens();
            step.run_chunk(p0, chunk)?;
            let scores = cohort_chunk_scores(&step, sequence, p0, baseline)
                .inspect_err(|_| step.state.poison())?;
            candidate.extend(scores);
        }
        if boundary(&step)
            != (Boundary {
                tokens: COHORT_CONTEXT,
                health: Health::Valid,
            })
        {
            return Err(bad(
                "cohort candidate did not end with valid complete state",
            ));
        }
    }
    step.stream
        .synchronize()
        .inspect_err(|_| step.state.poison())?;
    Ok(CohortRows {
        reference,
        candidate,
    })
}

fn cohort_deep_ids(ids: &[u32], case: DeepCase) -> Vec<u32> {
    ids.iter()
        .copied()
        .cycle()
        .take(case.prefix + DEEP_COUNT + 1)
        .collect()
}

fn cohort_deep_tail(
    step: &mut QwenResidentStep<'_>,
    sequence: &[u32],
    budget: usize,
    case: DeepCase,
    wide: bool,
) -> Result<CohortRows, EngError> {
    step.set_capture_config(CaptureConfig {
        capture: false,
        section_marks: false,
    })?;
    step.set_kv8_chunked_prefill(true)?;
    step.set_m_sess_cap(case.rows as u32);
    step.enable_pinned_prefix()?;
    let prefix_bytes = step.enable_device_tip()?;
    let verdict = step.admit_prefill(
        PrefillBudget {
            device_free: budget,
            extras: vec![super::super::prefill_plan::Extra {
                name: "device-prefix-delta",
                fixed: prefix_bytes,
                bytes_per_token: 0,
            }],
        },
        true,
        case.fixed,
        wide,
    )?;
    if verdict != (ChunkVerdict::Ready { m_sess: case.rows }) {
        return Err(bad("deep cohort did not admit the requested dispatch"));
    }
    let staging = step
        .prefill
        .as_ref()
        .ok_or_else(|| bad("deep cohort has no arena"))?
        .kv8_staging()?
        .ok_or_else(|| bad("deep cohort has no FP16 staging"))?;
    if staging.segmented != case.fixed {
        return Err(bad("deep cohort selected the wrong segmented mode"));
    }
    let staging = staging.plan;
    if case.fixed {
        let expected_stage = if wide {
            super::super::prefill_driver::FULL_STAGE_TIER_ROWS
        } else {
            super::super::prefill_driver::fixed_staging_rows(case.capacity)
        };
        if staging.capacity != expected_stage || staging.covers(case.prefix + DEEP_COUNT) {
            return Err(bad(
                "fixed deep cohort must exercise the bounded segmented stage",
            ));
        }
    } else if !staging.covers(case.prefix + DEEP_COUNT) {
        return Err(bad("deep cohort staging does not cover the scored tail"));
    }
    println!(
        "DEEP_COHORT_PATH prefix={} positions={DEEP_COUNT} m_sess={} staging_rows={} staging_bytes={} context_capacity={} prefix_delta_bytes={prefix_bytes} fixed_admission={} stored_kv=int8g32",
        case.prefix, case.rows, staging.capacity, staging.bytes, case.capacity, case.fixed
    );
    // Both scored arms restore this same committed prefix, regardless of its warm-up path.
    step.feed_rows_chunked(&sequence[..case.prefix], case.rows)?;
    let common = step.export_state()?;
    if common.tokens != case.prefix {
        return Err(bad("deep cohort common prefix has the wrong length"));
    }
    let mut reference = Vec::with_capacity(DEEP_COUNT);
    for pair in sequence[case.prefix..].windows(2) {
        step.run_token(pair[0], true)?;
        let logits = cohort_head_values(step.row_bytes())?;
        let scored = score(&logits, SAMPLER_VOCAB, pair[1] as usize).map_err(bad)?;
        reference.push(CohortReference {
            score: scored,
            logits,
        });
    }
    step.import_state(&common)?;
    if step.export_state()? != common || step.state.health() != Health::Valid {
        return Err(bad(
            "deep cohort arms do not share identical committed state",
        ));
    }
    drop(common);
    let segments_before = step.last_prefill_report().segmented_parts();
    let mut candidate = Vec::with_capacity(DEEP_COUNT);
    let mut offset = 0;
    while offset < DEEP_COUNT {
        let p0 = case.prefix + offset;
        let rows = case.rows.min(DEEP_COUNT - offset);
        let rows = if case.fixed {
            super::super::prefill_plan::fixed_depth_rows(p0, rows)
        } else {
            rows
        };
        let inputs = &sequence[p0..p0 + rows];
        println!(
            "DEEP_COHORT_DISPATCH p0={p0} rows={rows} arena_rows={}",
            case.rows
        );
        step.run_chunk(p0, inputs)?;
        for row in 0..inputs.len() {
            let baseline = reference
                .get(offset + row)
                .ok_or_else(|| bad("deep reference row is absent"))?;
            candidate.push(
                cohort_row_score(
                    step,
                    row,
                    sequence[case.prefix + offset + row + 1],
                    baseline,
                )
                .inspect_err(|_| step.state.poison())?,
            );
        }
        offset += rows;
    }
    step.stream
        .synchronize()
        .inspect_err(|_| step.state.poison())?;
    if boundary(step)
        != (Boundary {
            tokens: case.prefix + DEEP_COUNT,
            health: Health::Valid,
        })
    {
        return Err(bad("deep cohort did not commit every scored input"));
    }
    let parts = step
        .last_prefill_report()
        .segmented_parts()
        .checked_sub(segments_before)
        .ok_or_else(|| bad("deep cohort segmented counter moved backwards"))?;
    if case.fixed != (parts > 0) {
        return Err(bad(
            "deep cohort did not exercise the expected attention path",
        ));
    }
    println!("DEEP_COHORT_SEGMENT_PARTS {parts}");
    Ok(CohortRows {
        reference,
        candidate,
    })
}

#[test]
fn cohort_deep_span_has_exactly_1024_targets() {
    let ids = [3, 7, 11];
    for case in [DEEP_LEGACY, DEEP_FIXED] {
        let sequence = cohort_deep_ids(&ids, case);
        assert_eq!(sequence.len(), case.prefix + DEEP_COUNT + 1);
        let targets = &sequence[case.prefix + 1..];
        assert_eq!(targets.len(), 1024);
        for (offset, &target) in targets.iter().enumerate() {
            assert_eq!(target, ids[(case.prefix + offset + 1) % ids.len()]);
        }
        let positions: Vec<_> = (0..DEEP_COUNT)
            .step_by(case.rows)
            .flat_map(|first| first..first + case.rows)
            .collect();
        assert_eq!(positions, (0..DEEP_COUNT).collect::<Vec<_>>());
        assert!(cohort_deep_ids(&[], case).is_empty());
        assert!(sequence.len() <= case.capacity);
    }
}

#[test]
fn cohort_score_positions_match_both_oracle_chunks() {
    for rows in [16, 128, 512, 1024] {
        let positions: Vec<usize> = (0..COHORT_CONTEXT)
            .step_by(rows)
            .flat_map(|p0| cohort_positions(p0, rows).unwrap())
            .collect();
        assert_eq!(positions, (1024..2047).collect::<Vec<_>>());
        assert_eq!(positions.len() * 2, COHORT_COUNT);
    }
    assert!(cohort_positions(usize::MAX, 1).is_err());
    assert!(cohort_positions(2048, 1).is_err());
    assert!(cohort_positions(0, 0).is_err());
    assert!(cohort_ids(br#"{"bos":false,"ids":[]}"#).is_err());
    assert!(cohort_head_values(&[]).is_err());
    let uniform = [0.0, 0.0];
    let skewed = [0.8f32.ln(), 0.2f32.ln()];
    assert!(cohort_kl(&uniform, &uniform, 2).unwrap().abs() < 1e-12);
    assert!((cohort_kl(&skewed, &uniform, 2).unwrap() - 0.22314355131420976).abs() < 1e-7);
    assert!(cohort_kl(&uniform, &[f32::NAN, 0.0], 2).is_err());
    assert!(cohort_kl(&uniform, &uniform, 0).is_err());
}

fn run_cohort(chunk_rows: usize, deep: Option<DeepCase>) {
    let dir = std::env::var("Q36_CUDA_COHORT_DIR").unwrap_or_else(|_| {
        "/media/dan/nvme/fifi/scrch/bba24a15/llamacpp-oracle/cohort".to_owned()
    });
    let bytes = std::fs::read(std::path::Path::new(&dir).join("tokens.json"))
        .expect("read supplied oracle cohort without retokenizing");
    let ids = cohort_ids(&bytes).expect("unchanged supplied oracle cohort");
    let ids = if let Some(case) = deep {
        cohort_deep_ids(&ids, case)
    } else {
        ids
    };
    let observed =
        cohort_device_phase(&ids, chunk_rows, deep).expect("real native model cohort comparison");
    let (first, per_context, context, count) = if let Some(case) = deep {
        (
            case.prefix,
            DEEP_COUNT,
            case.prefix + DEEP_COUNT + 1,
            DEEP_COUNT,
        )
    } else {
        (COHORT_FIRST, 1023, COHORT_CONTEXT, COHORT_COUNT)
    };
    // No CUDA owners survive into numerical assertions.
    assert_eq!(observed.reference.len(), count);
    assert_eq!(observed.candidate.len(), count);
    let mut signed = 0.0;
    let mut kl_sum = 0.0;
    let mut same = 0usize;
    for (index, (reference, candidate)) in observed
        .reference
        .iter()
        .zip(&observed.candidate)
        .enumerate()
    {
        let chunk = index / per_context;
        let position = first + index % per_context;
        signed += candidate.score.nll - reference.score.nll;
        kl_sum += candidate.kl;
        same += usize::from(reference.score.top == candidate.score.top);
        println!(
            "COHORT_POSITION chunk={chunk} input={position} target={} reference_nll={:.17} candidate_nll={:.17} reference_top={} candidate_top={} kl_chunked_to_per_token={:.17}",
            chunk * context + position + 1,
            reference.score.nll,
            candidate.score.nll,
            reference.score.top,
            candidate.score.top,
            candidate.kl
        );
    }
    let mean = signed / count as f64;
    let agreement = same as f64 / count as f64;
    let mean_kl = kl_sum / count as f64;
    println!(
        "COHORT_GATE rows={chunk_rows} positions={count} mean_signed_delta_nll={mean:.12} top1_agreement={agreement:.12} mean_kl_chunked_to_per_token={mean_kl:.12} nll_limit=0.0117 top1_min=0.97"
    );
    assert!(mean <= 0.0117, "signed chunk-minus-production NLL {mean}");
    assert!(agreement >= 0.97, "top-1 agreement {agreement}");
    if deep.is_some() {
        assert!(mean_kl <= 0.004, "chunked-to-per-token KL {mean_kl}");
    }
}

#[test]
#[ignore = "full native model; supplied llama.cpp cohort; CPU/GPU locks required"]
fn kv8_chunk512_matches_production_on_oracle_cohort() {
    run_cohort(COHORT_CHUNK, None);
}

#[test]
#[ignore = "full native model; supplied llama.cpp cohort; CPU/GPU locks required"]
fn kv8_chunk1024_matches_production_on_oracle_cohort() {
    run_cohort(1024, None);
}

#[test]
#[ignore = "full native model above 64K; granted all-three-lock measurement window required"]
fn kv8_extended_staging_matches_production_above_64k() {
    run_cohort(DEEP_LEGACY.rows, Some(DEEP_LEGACY));
}

#[test]
#[ignore = "full native model at 192K; granted all-three-lock measurement window required"]
fn kv8_fixed_admission_matches_production_at_192k() {
    run_cohort(DEEP_FIXED.rows, Some(DEEP_FIXED));
}

#[test]
#[ignore = "full native model above 224K; granted all-three-lock measurement window required"]
fn kv8_segmented_depth256_matches_production_above_224k() {
    // Keep the 512-row arena while exercising the 256-row deep dispatch tier.
    run_cohort(
        DEEP_FIXED.rows,
        Some(DeepCase {
            prefix: 245_760,
            ..DEEP_FIXED
        }),
    );
}

#[test]
#[ignore = "full native model above the tier; split arms (WIDE_EQ_ARM=default|wide), logits via /task-tmp"]
fn wide_stage_matches_the_default_stage_above_the_tier() {
    // DEFAULT FLIP evidence: the same deep rig (180224 capacity, 171024 fed rows)
    // runs the default 32768-row stage and the wide 131072-row stage in separate
    // invocations (each fits the 900 s lease cap); the default arm dumps its
    // candidate logits to /task-tmp, the wide arm loads them, and the two arms
    // must pick identical tokens with a mean KL inside the deep cohort's own
    // chunked-vs-per-token bound.
    let arm = std::env::var("WIDE_EQ_ARM").unwrap_or_else(|_| "wide".to_owned());
    let run_wide = arm == "wide";
    let case = DeepCase {
        prefix: 170_000,
        capacity: 180_224,
        rows: 512,
        fixed: true,
    };
    let dir = std::env::var("Q36_CUDA_COHORT_DIR").unwrap_or_else(|_| {
        "/media/dan/nvme/fifi/scrch/bba24a15/llamacpp-oracle/cohort".to_owned()
    });
    let bytes =
        std::fs::read(std::path::Path::new(&dir).join("tokens.json")).expect("cohort tokens");
    let ids = cohort_ids(&bytes).expect("cohort ids");
    let ids = cohort_deep_ids(&ids, case);
    assert_eq!(ids.len(), case.prefix + DEEP_COUNT + 1);

    let dims = step_dims();
    let inv = inventory();
    let mut opts = options();
    opts.context_tokens = case.capacity;
    let rules = rules().expect("rules");
    let dir = std::env::var("Q36_CUDA_MODEL_DIR")
        .unwrap_or_else(|_| "/media/dan/nvme/models/qwen3.6-35b-a3b".to_owned());
    let model = open_qwen3_6_35b_a3b(
        &std::path::Path::new(&dir).join("Qwen3.6-35B-A3B-Q4_K_M.gguf"),
        None,
    )
    .expect("model");
    let projector = CudaProjector::new().expect("projector");
    let load = super::super::start::load_options(
        super::super::start::kv_cache_bytes(opts.context_tokens, opts.kv_mode).expect("kv bytes"),
        true,
    );
    let weights = projector.load_model(&model, load).expect("weights").device;
    let free = projector.memory_info().expect("free").free;
    assert!(
        free >= super::super::pinned::MIN_FREE_AFTER_LOAD,
        "equivalence rig is below the loader free floor"
    );
    let budget = free
        .checked_sub(super::super::start::prefill_headroom(true))
        .expect("equivalence rig has no admission headroom");
    let context = projector.context();
    let res: &'static StepResources = Box::leak(Box::new(
        StepResources::new(context, dims, &inv, &opts).expect("res"),
    ));
    let weights: &'static CudaWeights = Box::leak(Box::new(weights));
    let mut step =
        QwenResidentStep::new(context, res, weights, &opts, rules.clone()).expect("step");
    let observed = cohort_deep_tail(&mut step, &ids, budget, case, run_wide)
        .inspect_err(|_| step.state.poison())
        .expect("equivalence arm");
    assert_eq!(observed.candidate.len(), DEEP_COUNT);
    let dump = std::path::PathBuf::from(
        std::env::var("Q36_WIDE_EQ_DUMP")
            .unwrap_or_else(|_| "/task-tmp/wide_eq_logits.bin".to_owned()),
    );
    // Rig-only provenance: both arms build the identical string, so the
    // wide arm's header check compares the rig, never the arm that wrote it.
    let provenance = format!(
        "{{\"prefix\":{},\"capacity\":{},\"rows\":{}}}\n",
        case.prefix, case.capacity, case.rows
    );
    if !run_wide {
        // Arm 1: persist this arm's candidate logits for the wide arm to
        // compare, behind a provenance header the wide arm must match.
        let mut bytes = provenance.into_bytes();
        bytes.reserve(DEEP_COUNT * SAMPLER_VOCAB * 4);
        for c in &observed.candidate {
            for v in &c.logits {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        let len = bytes.len();
        std::fs::write(dump, bytes).expect("dump default-arm logits");
        println!("WIDE_EQ_DEFAULT_DUMP positions={} bytes={len}", DEEP_COUNT);
        return;
    }
    let raw = std::fs::read(dump).expect("default arm must run first (WIDE_EQ_ARM=default)");
    let split = raw
        .iter()
        .position(|&b| b == b'\n')
        .expect("dump provenance header");
    let (header, payload) = (&raw[..split], &raw[split + 1..]);
    assert_eq!(
        String::from_utf8_lossy(header),
        provenance.trim_end(),
        "dump provenance must match this rig"
    );
    let default_logits: Vec<f32> = payload
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(
        default_logits.len(),
        DEEP_COUNT * SAMPLER_VOCAB,
        "default-arm dump shape"
    );
    let mut same = 0usize;
    let mut kl_sum = 0.0;
    for (index, w) in observed.candidate.iter().enumerate() {
        let d = &default_logits[index * SAMPLER_VOCAB..(index + 1) * SAMPLER_VOCAB];
        let wt = w.score.top;
        let dt = d
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        same += usize::from(dt == wt);
        let kl = cohort_kl(d, &w.logits, SAMPLER_VOCAB).expect("kl");
        kl_sum += kl;
        println!(
            "WIDE_EQ_POSITION input={} default_top={dt} wide_top={wt} kl_default_to_wide={kl:.17}",
            case.prefix + index
        );
    }
    let agreement = same as f64 / DEEP_COUNT as f64;
    let mean_kl = kl_sum / DEEP_COUNT as f64;
    println!(
        "WIDE_EQ_GATE positions={DEEP_COUNT} top1_agreement={agreement:.12} mean_kl_default_to_wide={mean_kl:.12}"
    );
    assert_eq!(agreement, 1.0, "wide stage picked a different token");
    assert!(
        mean_kl <= 0.004,
        "wide-vs-default KL {mean_kl} exceeds the deep cohort bound"
    );
}
