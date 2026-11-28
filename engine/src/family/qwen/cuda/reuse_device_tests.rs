//! Prompt reuse between requests on the card step of a lineage model (`LINEAGE_STEP_GGUF`).
//!
//! The promise under test: a request served from a kept state has the bits of its cold start,
//! which is the same request, with the same boundary mark, served from an empty store. The
//! tests show the three facts that promise stands on (a cold start has the same bits twice, a
//! step reads no cache row at or above its position, the recorded steps survive a state put
//! back) and then the promise itself through the worker, the session and the served stepper.
//!
//! Needs a device and the file; without either a test skips with a note. One test per process
//! for a file that fills the card: every test loads the model.

use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::serve::{CardStep, load_model, model_of};
use super::step::{LoopStep, PREFILL_ROWS};
use crate::EngError;
use crate::family::executor::{GenerationState, SerialDecode};
use crate::family::project::Projector;
use crate::family::project::cuda::CudaWeights;
use crate::family::project::gguf::{GgufMatrix, GgufProjector};
use crate::family::sampler::Sampler;
use crate::family::session::{DecodeSession, Kept, TokenStepper, block_spacing_for};
use crate::family::step::tail::device_table;
use crate::kernels::cuda::{CudaContext, device_present};
use crate::sched::job::{Job, JobEvent};
use crate::sched::worker::{Engine, spawn_engine};
use crate::tokenizer::IncrementalDecoder;
use crate::tokenizer::gguf::bundle_from_lone_gguf;
use crate::weights::{GgufLimits, GgufWeights};

/// The tokens after the last message boundary of a test prompt.
const TAIL: usize = 4;
/// The tokens a served request decodes.
const DECODED: u32 = 8;

fn model_path(what: &str) -> Option<std::path::PathBuf> {
    let Ok(path) = std::env::var("LINEAGE_STEP_GGUF") else {
        eprintln!("skipped: set LINEAGE_STEP_GGUF");
        return None;
    };
    device_present(what).then(|| path.into())
}

fn open_file(path: &std::path::Path) -> Arc<GgufWeights> {
    let limits = GgufLimits {
        max_header_bytes: 32 << 20,
        max_tensor_bytes: 8 << 30,
    };
    Arc::new(GgufWeights::open_any_type(path, limits, None).unwrap())
}

/// `count` ids below `vocab`, a different stream for each `salt`.
fn ids(count: usize, salt: u32, vocab: usize) -> Vec<u32> {
    (0..count as u32)
        .map(|i| ((i ^ salt.wrapping_mul(0x9E37)).wrapping_mul(2_654_435_761) >> 7) % vocab as u32)
        .collect()
}

fn logits_bytes(card: &LoopStep<'_>) -> Vec<u8> {
    let logits = card.logits_row();
    let mut bytes = vec![0u8; logits.bytes()];
    logits.read_at(card.stream(), 0, &mut bytes).unwrap();
    bytes
}

/// The host side of a direct walk: the embedding rows of ids.
struct Rows {
    embd: GgufMatrix,
    projector: GgufProjector,
}

impl Rows {
    fn of(&mut self, tokens: &[u32]) -> Vec<f32> {
        self.projector.rows(&self.embd, tokens).unwrap()
    }
}

/// `prompt` from position 0 in pieces of [`PREFILL_ROWS`] rows, then `forced` as decode steps:
/// the logits row after every piece and after every step.
fn walk(card: &mut LoopStep<'_>, rows: &mut Rows, prompt: &[u32], forced: &[u32]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut pos = 0;
    for piece in prompt.chunks(PREFILL_ROWS) {
        card.prefill(&rows.of(piece), pos).unwrap();
        pos += piece.len();
        out.push(logits_bytes(card));
    }
    for &token in forced {
        card.decode(&rows.of(&[token]), pos).unwrap();
        pos += 1;
        out.push(logits_bytes(card));
    }
    out
}

fn first_difference(a: &[Vec<u8>], b: &[Vec<u8>]) -> Option<usize> {
    assert_eq!(a.len(), b.len(), "the walks have other lengths");
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// Two facts the reuse stands on, on one card. A cold start has the same bits twice: on a
/// second card, and on the first after a full reset. A step reads no cache row at or above its
/// position: the same walk has the same bits over planes that hold the rows of a longer
/// sequence, and over planes filled with NaN, where one read would spread through the logits.
#[test]
fn a_cold_start_repeats_its_bits_and_reads_no_cache_row_at_or_above_its_position() {
    const PROMPT: usize = 1500;
    const CACHE: usize = 2048;
    let Some(path) = model_path("the cold start of the card step") else {
        return;
    };
    let file = open_file(&path);
    let (model, routing) = model_of(&file).unwrap();
    let ctx = Rc::new(CudaContext::new().unwrap());
    let loaded = load_model(&ctx, &file, &model).unwrap();
    let (vocab, hidden) = (model.shape.vocab, model.shape.hidden);
    let mut rows = Rows {
        embd: GgufMatrix::new(file.header(), "token_embd.weight", vocab, hidden).unwrap(),
        projector: GgufProjector::new(file.clone()),
    };
    let open = || LoopStep::open_model(ctx.clone(), &loaded, &model, routing, CACHE).unwrap();
    let (prompt, forced) = (ids(PROMPT, 1, vocab - 300), ids(6, 2, vocab - 300));
    let mut card = open();
    let cold = walk(&mut card, &mut rows, &prompt, &forced);
    assert!(
        cold.iter().all(|row| row
            .as_chunks::<4>()
            .0
            .iter()
            .all(|w| f32::from_le_bytes(*w).is_finite())),
        "a cold row is not finite"
    );

    let other_card = walk(&mut open(), &mut rows, &prompt, &forced);
    assert_eq!(first_difference(&cold, &other_card), None, "a second card");
    card.reset().unwrap();
    let again = walk(&mut card, &mut rows, &prompt, &forced);
    assert_eq!(first_difference(&cold, &again), None, "after a full reset");

    // The planes keep the rows of a longer, other sequence; only the running state is cleared.
    card.reset().unwrap();
    walk(
        &mut card,
        &mut rows,
        &ids(CACHE - 8, 3, vocab - 300),
        &forced,
    );
    card.reset_running().unwrap();
    let over_rows = walk(&mut card, &mut rows, &prompt, &forced);
    assert_eq!(
        first_difference(&cold, &over_rows),
        None,
        "over the rows of another sequence"
    );

    // Every plane row is a half-precision NaN before the walk writes its own.
    card.fill_planes(0, 0x7E00u16.to_le_bytes()).unwrap();
    card.reset_running().unwrap();
    let over_nan = walk(&mut card, &mut rows, &prompt, &forced);
    assert_eq!(
        first_difference(&cold, &over_nan),
        None,
        "over planes of NaN"
    );
    eprintln!(
        "cold start: {} rows of {} bytes repeat on a second card, after a reset, over another \
         sequence's rows and over NaN planes",
        cold.len(),
        cold[0].len()
    );
}

/// The third fact: both recorded steps survive a state put back. After a prompt the state is
/// kept; decode steps record and replay; another sequence then runs from the same position; the
/// state is put back and the first steps run again as replays with the bits they had. The same
/// for the step from a token id, whose picks repeat as well, and a card that never kept
/// anything has those bits too.
#[test]
fn the_recorded_steps_survive_a_state_put_back() {
    const PROMPT: usize = 300;
    const STEPS: usize = 6;
    let Some(path) = model_path("the records across a put back") else {
        return;
    };
    let file = open_file(&path);
    let (model, routing) = model_of(&file).unwrap();
    let ctx = Rc::new(CudaContext::new().unwrap());
    let loaded = load_model(&ctx, &file, &model).unwrap();
    let (vocab, hidden) = (model.shape.vocab, model.shape.hidden);
    let mut rows = Rows {
        embd: GgufMatrix::new(file.header(), "token_embd.weight", vocab, hidden).unwrap(),
        projector: GgufProjector::new(file.clone()),
    };
    let open =
        || LoopStep::open_model(ctx.clone(), &loaded, &model, routing, PROMPT + 4 * STEPS).unwrap();
    let sampled = vocab - 300;
    let (prompt, forced, other) = (
        ids(PROMPT, 1, sampled),
        ids(STEPS, 2, sampled),
        ids(2 * STEPS, 3, sampled),
    );
    let fresh = walk(&mut open(), &mut rows, &prompt, &forced);
    let fresh = &fresh[1..];

    let mut card = open();
    card.prefill(&rows.of(&prompt), 0).unwrap();
    let kept: Kept = card.keep(PROMPT).unwrap();
    assert_eq!(kept.bytes(), card.kept_bytes());
    let steps = |card: &mut LoopStep<'_>, rows: &mut Rows, tokens: &[u32]| -> Vec<Vec<u8>> {
        tokens
            .iter()
            .enumerate()
            .map(|(at, &token)| {
                card.decode(&rows.of(&[token]), PROMPT + at).unwrap();
                logits_bytes(card)
            })
            .collect()
    };
    let first = steps(&mut card, &mut rows, &forced);
    assert!(card.replayed(), "the decode step never replayed its record");
    assert_eq!(first_difference(fresh, &first), None, "before a put back");
    steps(&mut card, &mut rows, &other);
    card.put_back(&kept).unwrap();
    card.decode(&rows.of(&forced[..1]), PROMPT).unwrap();
    assert!(card.replayed(), "the put back dropped the record");
    assert_eq!(logits_bytes(&card), first[0], "the replay after a put back");
    card.put_back(&kept).unwrap();
    let second = steps(&mut card, &mut rows, &forced);
    assert_eq!(first_difference(&first, &second), None, "after a put back");

    let Some(table) = device_table(&ctx, &file, "token_embd.weight", (vocab, hidden)).unwrap()
    else {
        eprintln!("no room for the embedding table: the token step was not checked");
        return;
    };
    card.set_embedding(table).unwrap();
    let token_steps = |card: &mut LoopStep<'_>, tokens: &[u32]| -> (Vec<u32>, Vec<Vec<u8>>) {
        tokens
            .iter()
            .enumerate()
            .map(|(at, &token)| {
                let picked = card.decode_token(token, PROMPT + at, sampled).unwrap();
                (picked, logits_bytes(card))
            })
            .unzip()
    };
    card.put_back(&kept).unwrap();
    let (picks, token_rows) = token_steps(&mut card, &forced);
    assert!(card.token_step_replayed(), "the token step never replayed");
    assert_eq!(
        first_difference(&first, &token_rows),
        None,
        "the token step"
    );
    token_steps(&mut card, &other);
    card.put_back(&kept).unwrap();
    let picked = card.decode_token(forced[0], PROMPT, sampled).unwrap();
    assert!(
        card.token_step_replayed(),
        "the put back dropped the record"
    );
    assert_eq!((picked, logits_bytes(&card)), (picks[0], first[0].clone()));
    card.put_back(&kept).unwrap();
    let (again, again_rows) = token_steps(&mut card, &forced);
    assert_eq!(again, picks, "the picks after a put back");
    assert_eq!(
        first_difference(&first, &again_rows),
        None,
        "the token rows"
    );
    eprintln!(
        "records: {STEPS} decode steps and {STEPS} token steps replay with their bits after a \
         put back of {} bytes",
        kept.bytes()
    );
}

/// What the served stepper ran for one request.
#[derive(Default, Clone, PartialEq)]
struct Ran {
    /// Every prompt chunk: its first position, its rows, and the logits row it left.
    chunks: Vec<(usize, usize, Vec<u8>)>,
    /// The logits row behind every decoded token, and the token.
    decoded: Vec<(Vec<u8>, u32)>,
    /// How many of the tokens were picked on the device.
    on_device: usize,
}

/// The served stepper with a record of what it ran. It forwards every call, so the session and
/// the worker drive the stepper they drive in a served engine.
struct Tap {
    card: CardStep,
    ran: Arc<Mutex<Ran>>,
}

impl TokenStepper for Tap {
    const OPERATION: &'static str = CardStep::OPERATION;

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        self.feed_rows(&[token])
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
        let first = self.card.fed();
        self.card.feed_rows(tokens)?;
        let row = self.card.last_logits()?;
        let mut ran = self.ran.lock().unwrap();
        ran.chunks.push((first, tokens.len(), row));
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        self.card.logits(token)
    }

    fn next_token(
        &mut self,
        token: u32,
        sampler: &mut Sampler,
        sampler_vocab: usize,
    ) -> Result<u32, EngError> {
        let on_device = self.card.picks_on_device()
            && sampler.is_greedy()
            && !sampler.capture_armed()
            && sampler.first_top2_margin().is_some();
        let picked = self.card.next_token(token, sampler, sampler_vocab)?;
        let row = self.card.last_logits()?;
        let mut ran = self.ran.lock().unwrap();
        ran.decoded.push((row, picked));
        ran.on_device += usize::from(on_device);
        Ok(picked)
    }

    fn fed(&self) -> usize {
        self.card.fed()
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.card.reset()
    }

    fn kept_bytes(&self) -> Option<u64> {
        self.card.kept_bytes()
    }

    fn keep(&mut self) -> Result<Kept, EngError> {
        self.card.keep()
    }

    fn put_back(&mut self, kept: &Kept) -> Result<(), EngError> {
        self.card.put_back(kept)
    }

    fn kept_generation(&self) -> u64 {
        self.card.kept_generation()
    }
}

/// One answer of the served engine: what the stepper ran, the tokens, the reused count, the
/// seconds to the first token and to the end.
struct Answer {
    ran: Ran,
    tokens: Vec<u32>,
    reused: usize,
    first_token: f64,
    whole: f64,
}

struct Served {
    engine: Engine,
    ran: Arc<Mutex<Ran>>,
    bundle: crate::tokenizer::TokenizerBundle,
}

impl Served {
    /// The card engine of `path` as `serve::start` builds it, with `cache` positions and the
    /// tap around its stepper.
    fn start(path: std::path::PathBuf, cache: usize, sampler_vocab: usize) -> Self {
        let ran = Arc::new(Mutex::new(Ran::default()));
        let thread_ran = Arc::clone(&ran);
        let bundle = bundle_from_lone_gguf(&path).unwrap();
        let engine = spawn_engine(PREFILL_ROWS, move |_| {
            let file = open_file(&path);
            let (model, routing) = model_of(&file)?;
            let ctx = Rc::new(CudaContext::new()?);
            let loaded: &'static CudaWeights =
                Box::leak(Box::new(load_model(&ctx, &file, &model)?));
            let step = LoopStep::open_model(Rc::clone(&ctx), loaded, &model, routing, cache)?;
            let tap = Tap {
                card: CardStep::new(step, &file, &model)?,
                ran: thread_ran,
            };
            let executor = SerialDecode::new(tap, Vec::new(), sampler_vocab, cache).with_reuse(
                PREFILL_ROWS,
                None,
                crate::family::step::keep::admit,
            );
            Ok((executor, DecodeSession::default()))
        })
        .unwrap();
        Self {
            engine,
            ran,
            bundle,
        }
    }

    /// Serves `prompt` greedily for [`DECODED`] tokens. With `marked`, its last message ends
    /// [`TAIL`] tokens before its end, as a chat request with reuse configured says.
    fn request(&self, prompt: &[u32], marked: bool) -> Answer {
        *self.ran.lock().unwrap() = Ran::default();
        let decoder = IncrementalDecoder::new(self.bundle.tokenizer());
        let generation = GenerationState::new(decoder).with_max_tokens(DECODED);
        let (mut job, mut handle) = Job::new(prompt.to_vec(), generation).unwrap();
        if marked {
            job = job.with_prefix_boundary(prompt.len() - TAIL).unwrap();
        }
        let began = Instant::now();
        assert!(
            self.engine.handle().try_submit(job).is_ok(),
            "the job queues"
        );
        let (mut tokens, mut reused, mut first_token) = (Vec::new(), None, None);
        while let Some(event) = handle.events.blocking_recv() {
            match event {
                JobEvent::Batch(batch) => {
                    first_token.get_or_insert_with(|| began.elapsed().as_secs_f64());
                    tokens.extend(batch.tokens);
                }
                JobEvent::Finished(usage) => reused = Some(usage.reused_tokens as usize),
                JobEvent::Error(error) => panic!("{error:?}"),
                JobEvent::Started => {}
            }
        }
        Answer {
            ran: self.ran.lock().unwrap().clone(),
            tokens,
            reused: reused.expect("the job finished"),
            first_token: first_token.expect("a token"),
            whole: began.elapsed().as_secs_f64(),
        }
    }

    /// A cold start of `prompt` on this engine: an unmarked request of another sequence first
    /// writes from position 0, after which no kept state matches, as with an empty store.
    fn cold(&self, prompt: &[u32], vocab: usize) -> Answer {
        self.request(&ids(3, 99, vocab), false);
        let answer = self.request(prompt, true);
        assert_eq!(answer.reused, 0, "a cold start reuses nothing");
        answer
    }
}

fn same_bits(a: &Ran, b: &Ran) -> bool {
    a.chunks == b.chunks && a.decoded == b.decoded
}

/// The promise, served: request 1, then request 2 = request 1, its completion and a new turn,
/// both with their boundary mark. Request 2 starts from the state kept at the last block
/// boundary below request 1's tip and feeds only the rows above it. Every prompt chunk it runs
/// is a chunk of its cold start, with the logits row of that chunk bit for bit; every decoded
/// token comes from the same logits row and is the same token, on the host sampler's path and
/// on the device pick. The cold start itself repeats bit for bit, also over the rows another
/// sequence left.
///
/// `REUSE_PROMPT_TOKENS` sets the length of request 1 (default 2,600). The seconds to the first
/// token are reported; the card is shared, so they describe this run only.
#[test]
fn a_reused_turn_has_the_bits_of_its_cold_start_with_the_same_boundary_mark() {
    let Some(path) = model_path("a reused turn on the served stepper") else {
        return;
    };
    let length: usize = std::env::var("REUSE_PROMPT_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2_600);
    let turn = 300;
    let vocab = bundle_from_lone_gguf(&path).unwrap().vocab_size() as usize;
    let cache = length + 2 * (turn + DECODED as usize) + 64;
    let served = Served::start(path, cache, vocab);

    let first = ids(length, 1, vocab);
    let one = served.request(&first, true);
    assert_eq!(one.reused, 0);
    assert_eq!(one.tokens.len(), DECODED as usize);
    let mut second = first.clone();
    second.extend(&one.tokens);
    second.extend(ids(turn, 2, vocab));

    let cold = served.cold(&second, vocab);
    let cold_again = served.cold(&second, vocab);
    assert!(
        same_bits(&cold.ran, &cold_again.ran),
        "a cold start has other bits the second time"
    );
    assert_eq!(cold.tokens, cold_again.tokens);

    let one_again = served.cold(&first, vocab);
    assert!(
        same_bits(&one.ran, &one_again.ran) && one.tokens == one_again.tokens,
        "request 1 has other bits over the rows of other sequences"
    );
    let warm = served.request(&second, true);
    let tip = first.len() - TAIL;
    let kept = (tip - 1) / block_spacing_for(tip, 4_096) * block_spacing_for(tip, 4_096);
    assert_eq!(
        warm.reused, kept,
        "the block boundary below request 1's tip"
    );
    assert_eq!(
        warm.ran.chunks.first().map(|chunk| chunk.0),
        Some(kept - 1),
        "the first row a reused request feeds"
    );
    let above: Vec<_> = cold
        .ran
        .chunks
        .iter()
        .filter(|chunk| chunk.0 + 1 >= kept)
        .cloned()
        .collect();
    assert!(!above.is_empty() && above.len() < cold.ran.chunks.len());
    assert!(
        warm.ran.chunks == above,
        "a reused request ran other chunks, or other bits, than its cold start"
    );
    assert!(
        warm.ran.decoded == cold.ran.decoded,
        "a reused request decoded from other bits than its cold start"
    );
    assert_eq!(warm.tokens, cold.tokens);
    assert_eq!(warm.ran.on_device, cold.ran.on_device);

    // A third turn starts from whatever cut request 2 left.
    let mut third = second.clone();
    third.extend(&warm.tokens);
    third.extend(ids(turn, 3, vocab));
    let warm_third = served.request(&third, true);
    assert!(warm_third.reused >= kept);
    let cold_third = served.cold(&third, vocab);
    let above: Vec<_> = cold_third
        .ran
        .chunks
        .iter()
        .filter(|chunk| chunk.0 + 1 >= warm_third.reused)
        .cloned()
        .collect();
    assert!(
        warm_third.ran.chunks == above && warm_third.ran.decoded == cold_third.ran.decoded,
        "the third turn has other bits than its cold start"
    );
    assert_eq!(warm_third.tokens, cold_third.tokens);

    let fed = |ran: &Ran| ran.chunks.iter().map(|chunk| chunk.1).sum::<usize>();
    let row_bytes = cold.ran.decoded[0].0.len();
    eprintln!(
        "reused turn: prompt {} tokens, kept at {kept}; rows fed cold {} reused {}; \
         {} chunk rows and {} decoded rows of {row_bytes} bytes equal; {} of {} tokens picked \
         on the device",
        second.len(),
        fed(&cold.ran),
        fed(&warm.ran),
        warm.ran.chunks.len(),
        warm.ran.decoded.len(),
        warm.ran.on_device,
        warm.tokens.len(),
    );
    eprintln!(
        "reused turn: first token after {:.3} s cold, {:.3} s cold again, {:.3} s reused; \
         whole request {:.3} s cold, {:.3} s reused (shared card)",
        cold.first_token, cold_again.first_token, warm.first_token, cold.whole, warm.whole,
    );
    eprintln!(
        "third turn: prompt {} tokens, reused {}; first token after {:.3} s cold, {:.3} s \
         reused",
        third.len(),
        warm_third.reused,
        cold_third.first_token,
        warm_third.first_token,
    );
    drop(served.engine.shutdown());
}
