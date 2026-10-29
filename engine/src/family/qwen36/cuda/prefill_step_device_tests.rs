//! Device rows of the chunked-prefill driver, one test per process on the 3090, held exclusively.
//! They load the pinned 20 GB model, so they are ignored by default and run by name. The rows cover
//! a clean chunk, continuation at the committed rows, an injected router fault, a stale prefill
//! word, the token ids of the chunk rows, an unadmitted session and the worker's pieces.

use super::book::Health;
use super::common_tests::{dims as step_dims, inventory};
use super::dims::{KvMode, StepOptions};
use super::prefill_driver::PrefillBudget;
use super::resources::StepResources;
use super::rules::QwenRules;
use super::step::QwenResidentStep;
use crate::EngError;
use crate::family::gguf_model::open_qwen3_6_35b_a3b;
use crate::family::project::cuda::{
    CudaProjector, CudaWeights, LoadOptions, Reserve, WeightLayout,
};
use crate::family::session::TokenStepper;
use serde_json::Value;

const SAMPLER_VOCAB: usize = 248_077;
const HEAD_ROWS: usize = 248_320;
const EOS: [u32; 2] = [248_046, 248_044];
const CAP: u32 = 128;
const CONTEXT_TOKENS: usize = 1024;

fn model_dir() -> String {
    std::env::var("Q36_CUDA_MODEL_DIR")
        .unwrap_or_else(|_| "/media/dan/nvme/models/qwen3.6-35b-a3b".to_string())
}

fn prompt_ids(name: &str) -> Vec<u32> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../qualification/cuda-rd/oracle/prompts/")
        .join(format!("{name}.ids.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let value: Value = serde_json::from_str(&text).expect("ids json");
    value["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|id| id.as_u64().expect("id") as u32)
        .collect()
}

fn rules() -> QwenRules {
    QwenRules::new(EOS.to_vec(), SAMPLER_VOCAB, HEAD_ROWS).expect("rules")
}

fn options() -> StepOptions {
    StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: CONTEXT_TOKENS,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    }
}

fn load() -> Result<(CudaProjector, CudaWeights), EngError> {
    let gguf_path = std::path::Path::new(&model_dir()).join("Qwen3.6-35B-A3B-Q4_K_M.gguf");
    let model = open_qwen3_6_35b_a3b(&gguf_path, None)?;
    let projector = CudaProjector::new()?;
    let device = projector
        .load_model(
            &model,
            LoadOptions {
                reserve: Reserve::default(),
                drop_page_cache: None,
                layout: WeightLayout::Native,
            },
        )?
        .device;
    Ok((projector, device))
}

/// One observed chunk run, all host values: nothing here borrows the device.
struct Observed {
    tokens: usize,
    steps: Vec<(usize, usize, i32)>,
    n_prefilled: usize,
    n_fed: usize,
    arena_bytes: usize,
    part_bytes: usize,
    logits_finite: usize,
    logits_len: usize,
    logits_bytes: Vec<u8>,
}

fn build(
    projector: CudaProjector,
    weights: CudaWeights,
) -> Result<(QwenResidentStep<'static>, usize), EngError> {
    // The owners are leaked exactly as make() does, so the step can outlive this function.
    let free = projector.memory_info()?.free;
    let context = projector.context();
    let dims = step_dims();
    let inv = inventory();
    let opts = options();
    let res: &'static StepResources =
        Box::leak(Box::new(StepResources::new(context, dims, &inv, &opts)?));
    let projector: &'static CudaProjector = Box::leak(Box::new(projector));
    let weights: &'static CudaWeights = Box::leak(Box::new(weights));
    let step = QwenResidentStep::new(projector.context(), res, weights, &opts, rules())?;
    Ok((step, free))
}

/// Admits the arena from the free VRAM measured after the load, as make() does, and answers the
/// verdict as text.
fn admit(step: &mut QwenResidentStep<'static>, free: usize) -> Result<String, EngError> {
    step.set_m_sess_cap(CAP);
    let verdict = step.admit_prefill(
        PrefillBudget {
            device_free: free.saturating_sub(1usize << 30),
            extras: Vec::new(),
        },
        false,
        false,
        false,
    )?;
    Ok(format!("{verdict:?}"))
}

/// Runs one chunked call and the handoff decode, and returns the host values.
fn observe(
    step: &mut QwenResidentStep<'static>,
    head: &[u32],
    last: u32,
) -> Result<Observed, EngError> {
    step.feed_rows_chunked(head, usize::try_from(CAP).unwrap())?;
    let logits = step.logits(last)?;
    let report = step.last_prefill_report();
    let (_, arena_bytes, part_bytes, _, _) = report.statics();
    Ok(Observed {
        tokens: step.tokens(),
        steps: report
            .steps()
            .iter()
            .map(|s| (s.p0, s.rows, s.status))
            .collect(),
        n_prefilled: report.n_prefilled(),
        n_fed: report.n_fed(),
        arena_bytes,
        part_bytes,
        logits_finite: logits.iter().filter(|v| v.is_finite()).count(),
        logits_len: logits.len(),
        logits_bytes: step.row_bytes().to_vec(),
    })
}

/// The step's own stream, for the one direct device write of the stale-word row.
fn step_stream<'s>(step: &'s QwenResidentStep<'static>) -> &'s crate::kernels::cuda::DeviceStream {
    step.stream_for_tests()
}

#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn a_router_fault_in_a_chunk_poisons_and_publishes_nothing() {
    let p5 = prompt_ids("P5");
    let (projector, weights) = load().expect("load");
    let (mut step, free) = build(projector, weights).expect("step");
    let verdict = admit(&mut step, free).expect("admit");
    assert!(verdict.starts_with("Ready"), "{verdict}");
    for &token in &p5[..4] {
        step.feed(token).expect("four single rows");
    }
    assert_eq!(step.tokens(), 4);

    // The real bit-1 fault: a NaN over layer 0's recurrent range reaches the router, whose fault
    // entry ORs bit 1 into the prefill word.
    let stream = step_stream(&step);
    {
        let range = step
            .state_for_tests()
            .delta_rec(0, 0)
            .expect("layer 0 recurrent range");
        let nan = f32::NAN.to_le_bytes();
        let mut bytes = Vec::with_capacity(range.bytes());
        while bytes.len() < range.bytes() {
            bytes.extend_from_slice(&nan);
        }
        bytes.truncate(range.bytes());
        range
            .buffer()
            .write_at(stream, range.offset(), &bytes)
            .expect("NaN write");
    }

    let error = step
        .feed_rows_chunked(&p5[4..140], usize::try_from(CAP).unwrap())
        .expect_err("the faulty chunk must fail");
    let recorded: Vec<(usize, usize, i32)> = step
        .last_prefill_report()
        .steps()
        .iter()
        .map(|s| (s.p0, s.rows, s.status))
        .collect();
    let health = step.state_for_tests().health();
    let tokens = step.tokens();
    let refused = step.logits(p5[140]).is_err();
    step.reset_state().expect("reset");
    let after_reset = (step.state_for_tests().health(), step.tokens());
    let clean = observe(&mut step, &p5[..130], p5[130]);

    assert!(
        matches!(&error, EngError::Backend { operation, .. } if *operation == "route_experts"),
        "{error:?}"
    );
    assert_eq!(
        recorded,
        vec![(4, 128, 1)],
        "the record is pushed before the state is revalidated, with status 1"
    );
    assert!(
        format!("{health:?}").contains("Poisoned"),
        "the chunk poisons the session: {health:?}"
    );
    assert_eq!(tokens, 4, "a failed chunk publishes nothing");
    assert!(refused, "a poisoned session refuses the next token");
    assert_eq!(
        format!("{:?}", after_reset.0),
        format!("{:?}", Health::Valid),
        "a reset returns the session to Valid"
    );
    assert_eq!(after_reset.1, 0);
    let clean = clean.expect("a clean chunk publishes after the reset");
    assert_eq!(
        clean.steps,
        vec![(0, 128, 0), (128, 2, 0)],
        "130 rows are a chunk and a two-row chunk; the clean run is unaffected"
    );
}

#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn a_stale_prefill_word_cannot_fail_a_clean_chunk() {
    use super::state_layout::Word;

    let p5 = prompt_ids("P5");
    let head = &p5[..129];
    let (projector, weights) = load().expect("load");
    let (mut step, free) = build(projector, weights).expect("step");
    let verdict = admit(&mut step, free).expect("admit");
    assert!(verdict.starts_with("Ready"), "{verdict}");
    // A stale 1 in the status word is exactly what a previous chunk could leave behind.
    let word = step
        .state_for_tests()
        .word(Word::PrefillStatus)
        .expect("word range");
    word.buffer()
        .write_at(step_stream(&step), word.offset(), &1i32.to_le_bytes())
        .expect("stale write");
    let observed = observe(&mut step, head, p5[129]).expect("clean chunk");
    assert_eq!(
        observed.steps,
        vec![(0, 128, 0), (128, 1, 0)],
        "the clear makes the stale word harmless"
    );
    assert_eq!(observed.n_prefilled, 128);
    assert_eq!(observed.n_fed, 1, "the 129-row prompt ends in one feed row");
}

#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn chunk_rows_carry_their_token_ids() {
    let p5 = prompt_ids("P5");
    let p1 = prompt_ids("P1");
    let mut a: Vec<u32> = p5[..129].to_vec();
    let last = p5[129];
    let mut b = a.clone();
    b[..128].copy_from_slice(&p1[..128]);
    a.pop();
    b.pop();

    let (projector, weights) = load().expect("load");
    let (mut step, free) = build(projector, weights).expect("step");
    let verdict = admit(&mut step, free).expect("admit");
    assert!(verdict.starts_with("Ready"), "{verdict}");
    let first = observe(&mut step, &a, last).expect("A");
    step.reset_state().expect("reset");
    let again = observe(&mut step, &a, last).expect("A again");
    step.reset_state().expect("reset");
    let other = observe(&mut step, &b, last).expect("B");

    assert_eq!(
        first.logits_finite, first.logits_len,
        "every logit finite in A"
    );
    assert_eq!(
        other.logits_finite, other.logits_len,
        "every logit finite in B"
    );
    assert_eq!(first.tokens, 129);
    assert_eq!(other.tokens, 129);
    assert_eq!(
        first.logits_bytes, again.logits_bytes,
        "A twice is bit-identical"
    );
    assert_ne!(
        first.logits_bytes, other.logits_bytes,
        "different leading ids change the head row"
    );
}

#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn an_unadmitted_session_keeps_the_per_token_path() {
    use super::run_report::{PathTag, StepKind};
    use super::step::ChunkVerdict;
    use crate::family::executor::ExecutorOps;

    let p5 = prompt_ids("P5");
    let (projector, weights) = load().expect("load");
    let (mut step, _free) = build(projector, weights).expect("step");
    step.set_m_sess_cap(CAP);
    let verdict = step
        .admit_prefill(
            PrefillBudget {
                device_free: 0,
                extras: Vec::new(),
            },
            false,
            false,
            false,
        )
        .expect("a refused admission is not an error");
    let line = super::start::prefill_start_line(&verdict);
    let piece = ExecutorOps::prefill_piece(&step);
    step.feed_rows(&p5[..20]).expect("per-token feed");
    let report = step.last_prefill_report();
    let path = report.path_taken();
    let n_fed = report.n_fed();
    let all_feeds = report.steps().iter().all(|s| s.kind == StepKind::Feed);
    let steps = report.steps().len();
    let health = step.state_for_tests().health();
    let tokens = step.tokens();

    assert!(matches!(verdict, ChunkVerdict::NoArena), "{verdict:?}");
    assert_eq!(line, "q36 cuda: prefill per token: no arena");
    assert_eq!(piece, None, "an unadmitted session answers no piece");
    assert_eq!(
        format!("{health:?}"),
        format!("{:?}", Health::Valid),
        "the session is healthy before the report asserts"
    );
    assert_eq!(tokens, 20);
    assert_eq!(
        format!("{path:?}"),
        format!("{:?}", Some(PathTag::PerToken))
    );
    assert_eq!(n_fed, 20, "every row went through the per-token feed");
    assert_eq!(steps, 20, "one feed step per token");
    assert!(all_feeds, "no chunk step was recorded");
}

/// What the observer records per prefill call: the tokens it received and the pieces the
/// report held afterwards.
type SeenCalls = std::sync::Arc<std::sync::Mutex<Vec<(usize, Vec<usize>)>>>;

/// The engine-side observer of `the_worker_feeds_m_sess_tokens_per_piece`: it delegates every
/// ExecutorOps method and records, after each prefill call, the tokens it received and the pieces
/// the report had recorded. Test only.
struct PieceObserver {
    inner: QwenResidentStep<'static>,
    seen: SeenCalls,
}

impl crate::family::executor::ExecutorOps for PieceObserver {
    type Session = crate::family::session::DecodeSession;

    fn history<'s>(&self, st: &'s Self::Session) -> &'s [u32] {
        crate::family::executor::ExecutorOps::history(&self.inner, st)
    }

    fn reset(&mut self, st: &mut Self::Session) -> Result<(), EngError> {
        crate::family::executor::ExecutorOps::reset(&mut self.inner, st)
    }

    fn prefill(&mut self, st: &mut Self::Session, toks: &[u32]) -> Result<(), EngError> {
        let out = crate::family::executor::ExecutorOps::prefill(&mut self.inner, st, toks);
        let pieces: Vec<usize> = self.inner.last_prefill_report().piece_tokens().to_vec();
        self.seen.lock().expect("seen").push((toks.len(), pieces));
        out
    }

    fn decode(
        &mut self,
        st: &mut Self::Session,
        generation: &mut crate::family::executor::GenerationState,
    ) -> Result<crate::tokenizer::DecodeBatch, EngError> {
        crate::family::executor::ExecutorOps::decode(&mut self.inner, st, generation)
    }

    fn boundary(&mut self, st: &mut Self::Session) {
        crate::family::executor::ExecutorOps::boundary(&mut self.inner, st);
    }

    fn prefill_piece(&self) -> Option<usize> {
        crate::family::executor::ExecutorOps::prefill_piece(&self.inner)
    }
}

#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn the_worker_feeds_m_sess_tokens_per_piece() {
    use crate::family::executor::GenerationState;
    use crate::family::session::DecodeSession;
    use crate::sched::job::{Job, JobEvent};
    use crate::sched::worker::{EngineState, spawn_engine};
    use crate::tokenizer::IncrementalDecoder;
    use crate::tokenizer::Tokenizer;
    use std::sync::{Arc, Mutex};

    let p1 = prompt_ids("P1");
    let prompt: Vec<u32> = p1[..300].to_vec();
    let seen: SeenCalls = Arc::new(Mutex::new(Vec::new()));
    let seen_in = Arc::clone(&seen);

    let engine = spawn_engine(16, move || {
        // The model is loaded INSIDE the engine thread: no CUDA value crosses a thread.
        let (projector, weights) = load().expect("load on the engine thread");
        let (mut step, free) = build(projector, weights).expect("step");
        let verdict = admit(&mut step, free).expect("admit");
        assert!(verdict.starts_with("Ready"), "{verdict}");
        Ok((
            PieceObserver {
                inner: step,
                seen: seen_in,
            },
            DecodeSession::default(),
        ))
    })
    .expect("spawn");
    let handle = engine.handle();

    let tokenizer_path = std::path::Path::new(&model_dir()).join("tokenizer.json");
    let tokenizer_bytes = std::fs::read(&tokenizer_path).expect("tokenizer.json beside the gguf");
    // The real tokenizer: a synthetic three-token one cannot decode model ids.
    let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes).expect("parse tokenizer");
    let generation = GenerationState::new(IncrementalDecoder::new(&tokenizer)).with_max_tokens(1);
    let (job, mut events) = Job::new(prompt, generation).expect("job");
    handle.try_submit(job).expect("submit");
    let mut errors = 0;
    let mut finished = 0;
    while let Some(event) = events.events.blocking_recv() {
        match event {
            JobEvent::Error(ref e) => {
                errors += 1;
                eprintln!("JOB_ERROR {e:?}");
            }
            JobEvent::Finished(_) => {
                finished += 1;
                break;
            }
            JobEvent::Batch(_) => {}
        }
    }
    let report_state = engine.shutdown().state;
    let calls: Vec<usize> = seen.lock().expect("seen").iter().map(|(n, _)| *n).collect();
    let last_pieces: Vec<usize> = seen
        .lock()
        .expect("seen")
        .last()
        .map(|(_, pieces)| pieces.clone())
        .unwrap_or_default();

    assert_eq!(errors, 0, "the job finished without an error event");
    assert_eq!(finished, 1);
    assert_eq!(report_state, EngineState::Stopped);
    assert_eq!(
        calls,
        vec![128, 128, 44],
        "the worker feeds M_sess tokens per piece, three pieces for 300 tokens"
    );
    assert_eq!(
        last_pieces,
        vec![128, 128, 44],
        "the report records one piece per call"
    );
}

#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn a_clean_chunk_publishes_and_hands_off_to_decode() {
    let p5 = prompt_ids("P5");
    let head = &p5[..200];
    let last = p5[200];
    let (projector, weights) = load().expect("load");
    let (mut step, free) = build(projector, weights).expect("step");
    let verdict = admit(&mut step, free).expect("admit");
    let first = observe(&mut step, head, last).expect("run 1");
    step.reset_state().expect("reset");
    let second = observe(&mut step, head, last).expect("run 2");

    assert!(verdict.starts_with("Ready"), "{verdict}");
    assert_eq!(
        first.steps,
        vec![(0, 128, 0), (128, 72, 0)],
        "two clean chunks"
    );
    assert_eq!(first.n_prefilled, 200);
    assert_eq!(first.n_fed, 0, "a 200-row prompt has no one-row step");
    assert_eq!(
        first.tokens, 201,
        "the handoff decode publishes the last row"
    );
    assert_eq!(first.logits_finite, first.logits_len, "every logit finite");
    assert_eq!(first.logits_len, HEAD_ROWS);
    assert!(first.arena_bytes > 0 && first.part_bytes > 0);
    assert_eq!(
        first.logits_bytes, second.logits_bytes,
        "a fixed M, prompt and p0 are bit-identical after a reset"
    );
}

#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn chunks_continue_at_the_committed_rows() {
    let p5 = prompt_ids("P5");
    let (projector, weights) = load().expect("load");
    let (mut step, free) = build(projector, weights).expect("step");
    step.set_m_sess_cap(CAP);
    let verdict = admit(&mut step, free).expect("admit");
    assert!(verdict.starts_with("Ready"), "{verdict}");
    for &token in &p5[..3] {
        step.feed(token).expect("three single rows");
    }
    assert_eq!(step.tokens(), 3);
    step.feed_rows_chunked(&p5[3..132], usize::try_from(CAP).unwrap())
        .expect("first chunked call");
    let after_first: Vec<(usize, usize, i32)> = step
        .last_prefill_report()
        .steps()
        .iter()
        .map(|s| (s.p0, s.rows, s.status))
        .collect();
    let tokens_first = step.tokens();
    let start_rows = step.last_prefill_report().start_rows();
    step.feed_rows_chunked(&p5[132..202], usize::try_from(CAP).unwrap())
        .expect("second chunked call");
    let after_second: Vec<(usize, usize, i32)> = step
        .last_prefill_report()
        .steps()
        .iter()
        .map(|s| (s.p0, s.rows, s.status))
        .collect();
    let tokens_second = step.tokens();
    let logits = step.logits(p5[202]).expect("handoff decode");
    let finite = logits.iter().filter(|v| v.is_finite()).count();
    let len = logits.len();

    assert_eq!(
        tokens_first, 132,
        "the first call publishes its chunk and tail"
    );
    assert_eq!(tokens_second, 202);
    assert_eq!(
        start_rows, 3,
        "start_rows is the epoch's first chunked position"
    );
    assert_eq!(
        after_first,
        vec![(3, 128, 0), (131, 1, 0)],
        "chunk then one-row tail"
    );
    assert_eq!(after_second, vec![(3, 128, 0), (131, 1, 0), (132, 70, 0)]);
    assert_eq!(finite, len, "every logit finite");
    step.reset_state().expect("reset");
    assert!(
        step.last_prefill_report().steps().is_empty(),
        "a reset empties the report"
    );
    assert_eq!(step.tokens(), 0);
}

#[test]
#[ignore = "the 3090 by name (CPF wide stage): loads the real model; run by name"]
fn wide_stage_admits_under_the_real_card_free() {
    use super::step::ChunkVerdict;

    // The live-free snapshot gates BOTH candidates inside the selection. This pin
    // builds the cohort rig (Int8G32 fixed admission with the pinned prefix — the
    // only configuration where the wide candidate exists) at 180224 context: the
    // post-load free there has room for the 256 MiB wide stage, so the wide flag
    // must admit it rather than refuse a boot the default keeps. At 262144 the
    // fully-resident state leaves even the default stage over budget, which the
    // fallback pins cover host-side.
    let dir = std::env::var("Q36_CUDA_MODEL_DIR")
        .unwrap_or_else(|_| "/media/dan/nvme/models/qwen3.6-35b-a3b".to_owned());
    let model = open_qwen3_6_35b_a3b(
        &std::path::Path::new(&dir).join("Qwen3.6-35B-A3B-Q4_K_M.gguf"),
        None,
    )
    .expect("model");
    let projector = CudaProjector::new().expect("projector");
    let opts = StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: 180_224,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    };
    let kv_bytes =
        super::start::kv_cache_bytes(opts.context_tokens, opts.kv_mode).expect("kv cache bytes");
    let weights = projector
        .load_model(&model, super::start::load_options(kv_bytes, true))
        .expect("fixed load")
        .device;
    let free = projector.memory_info().expect("memory info").free;
    assert!(
        free >= super::pinned::MIN_FREE_AFTER_LOAD,
        "the pin needs the loader free floor"
    );
    let budget = free
        .checked_sub(super::start::prefill_headroom(true))
        .expect("the pin needs admission headroom");
    let context = projector.context();
    let dims = step_dims();
    let inv = inventory();
    let res: &'static StepResources = Box::leak(Box::new(
        StepResources::new(context, dims, &inv, &opts).expect("res"),
    ));
    let projector: &'static CudaProjector = Box::leak(Box::new(projector));
    let weights: &'static CudaWeights = Box::leak(Box::new(weights));
    let mut step =
        QwenResidentStep::new(projector.context(), res, weights, &opts, rules()).expect("step");
    step.set_kv8_chunked_prefill(true).expect("chunked");
    // Production input cap: fixed_chunk_rows then selects the 512-row arena the
    // shipping fixed path and the deep cohort use, not the file-wide test CAP.
    step.set_m_sess_cap(1024);
    // Mirror the cohort rig: the pinned prefix changes the KV state layout, and every
    // real boot that can offer the wide stage runs with it enabled.
    step.enable_pinned_prefix().expect("pinned prefix");
    let prefix_bytes = step.enable_device_tip().expect("device tip");
    assert!(
        prefix_bytes > 0,
        "device tip must not fall back to the host"
    );
    let verdict = step
        .admit_prefill(
            PrefillBudget {
                device_free: budget,
                extras: vec![super::prefill_plan::Extra {
                    name: "device-prefix-delta",
                    fixed: prefix_bytes,
                    bytes_per_token: 0,
                }],
            },
            true,
            true,
            true,
        )
        .expect("the wide flag must not refuse the boot");
    assert_eq!(
        verdict,
        ChunkVerdict::Ready { m_sess: 512 },
        "the production 512-row fixed arena must win the wide stage"
    );
    let plan = step
        .kv8_staging_plan_for_tests()
        .expect("fixed KV8 staging is attached");
    assert_eq!(
        plan.capacity,
        super::prefill_driver::FULL_STAGE_TIER_ROWS,
        "the wide stage must win under the real card free"
    );
}
