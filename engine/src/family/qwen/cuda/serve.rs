//! Serves a model of the lineage from the card step: a stepper over [`LoopStep`] for the shared
//! serial decode session. The file's header says which model it is; the layer list, the device
//! load and the step follow from that, so every model of the lineage starts here.
//!
//! The engine thread builds everything CUDA inside the worker's `make`. A request's prompt rows
//! take the prefill in pieces of up to [`PREFILL_ROWS`] rows; the pending token and every drawn
//! token take one decode step, and the head's row is read back for the host sampler. The
//! embedding row is gathered on the host: the table is not part of the device load.
//!
//! Lifetime: the worker needs an executor without a lifetime and the step borrows the device
//! load, so the load is `Box::leak`ed once per engine. ONE card engine is built per process: a
//! guard takes the flag before the first leak, and a retry in the same process is refused.

use std::{
    path::PathBuf,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use super::step::{LoopStep, PREFILL_ROWS, Routing, looped};
use crate::EngError;
use crate::family::executor::SerialDecode;
use crate::family::mem::DEVICE;
use crate::family::project::Projector;
use crate::family::project::cuda::{
    BorrowedLoader, CudaWeights, LoadOptions, WeightLayout, served_admission,
};
use crate::family::project::gguf::{GgufMatrix, GgufProjector};
use crate::family::qwen::recipe::{Model, ROUTED_EXPERTS, layers, routed_experts, tensors};
use crate::family::qwen::sensenova::weights::LoopWeights;
use crate::family::sampler::Sampler;
use crate::family::session::{DecodeSession, Kept, TokenStepper, sample_logits};
use crate::family::step::tail::device_table;
use crate::kernels::cuda::CudaContext;
use crate::kernels::cuda::resident::embed::EmbedTable;
use crate::sched::worker::{Engine, ReadyNote, spawn_engine_noted};
use crate::weights::{GgufLimits, GgufWeights, MetaValue};

/// The model of the lineage a file holds and the routing of its two-stream range: the
/// routed-expert model by its architecture name, the looped model otherwise (its own reader
/// refuses a header that is neither).
pub(crate) fn model_of(file: &Arc<GgufWeights>) -> Result<(Model, Routing), EngError> {
    let routed = MetaValue::String(ROUTED_EXPERTS.to_owned());
    if file.header().meta("general.architecture") == Some(&routed) {
        // One residual stream: the routing of a two-stream range is unused.
        let routing = Routing {
            pre_scale: 1.0,
            iterations: 1,
            fixed_identity: true,
        };
        return Ok((routed_experts(file.header())?, routing));
    }
    let host = LoopWeights::open(Arc::clone(file))?;
    Ok((
        looped(host.shape(), host.plan(), host.rope()),
        host.plan().routing(),
    ))
}

/// The device load of `model`: every tensor its layers name, in the blocks of the file.
pub(crate) fn load_model(
    ctx: &Rc<CudaContext>,
    file: &GgufWeights,
    model: &Model,
) -> Result<CudaWeights, EngError> {
    let names = tensors(&layers(&model.visits, model.blocks).0);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    BorrowedLoader::new(ctx)?.load_sliced(
        file,
        &refs,
        LoadOptions {
            layout: WeightLayout::Native,
            ..LoadOptions::default()
        },
        &[],
        0,
    )
}

/// Starts the card engine of one GGUF file and returns it with the context it serves: the
/// model's `trained` context, cut to the positions the device admits once the weights and the
/// step's scratch are on it. The engine thread announces that number as soon as it is planned.
pub(crate) fn start(
    file: PathBuf,
    limits: GgufLimits,
    ready_note: ReadyNote,
    eos: Vec<u32>,
    sampler_vocab: usize,
    trained: u32,
) -> Result<(Engine, u32), EngError> {
    let mut guard = BuildGuard::acquire()?;
    let (announce, resolved) = std::sync::mpsc::sync_channel::<Result<u32, String>>(1);
    let engine = spawn_engine_noted(PREFILL_ROWS, ready_note, move |_| {
        let began = Instant::now();
        let opened = (|| -> Result<(CardStep, usize, &'static str), EngError> {
            let ctx = Rc::new(CudaContext::with_admission(&served_admission())?);
            let gguf = Arc::new(GgufWeights::open_any_type(&file, limits, None)?);
            let (model, routing) = model_of(&gguf)?;
            let loaded = load_model(&ctx, &gguf, &model)?;
            // Past this point the load is the process's: see the module note.
            guard.commit();
            let loaded: &'static CudaWeights = Box::leak(Box::new(loaded));
            // What the step still takes after its cache is sized: the embedding table of the
            // token tail (when the device gather reads its stored type) and the allocations a
            // serving process makes after it measures. If keeping room for the table leaves too
            // little cache, fall back to host embeddings rather than refusing to start.
            let table = gguf
                .header()
                .tensor("token_embd.weight")
                .filter(|tensor| EmbedTable::gathers(tensor.ggml_type()))
                .map_or(0, |tensor| tensor.byte_len());
            let base_free = DEVICE
                .runtime_overhead_floor
                .max(DEVICE.embedding_table_reserve);
            let step = match LoopStep::open_planned(
                Rc::clone(&ctx),
                loaded,
                &model,
                routing,
                trained as usize,
                base_free.saturating_add(table),
            ) {
                Ok(step) => step,
                Err(EngError::MemoryPlan(_)) if table > 0 => LoopStep::open_planned(
                    Rc::clone(&ctx),
                    loaded,
                    &model,
                    routing,
                    trained as usize,
                    DEVICE.runtime_overhead_floor,
                )?,
                Err(error) => return Err(error),
            };
            let rows = step.cache_rows();
            Ok((CardStep::new(step, &gguf, &model)?, rows, model.family))
        })();
        let (card, rows, family) = match opened {
            Ok(opened) => opened,
            Err(error) => {
                let _ = announce.send(Err(error.describe()));
                return Err(error);
            }
        };
        let _ = announce.send(Ok(u32::try_from(rows).unwrap_or(u32::MAX)));
        eprintln!(
            "{family} cuda: ready in {:.1} s; the card step serves {rows} positions",
            began.elapsed().as_secs_f64()
        );
        // An entry whose configuration marks no prompt boundary never reaches the kept states.
        let budget =
            crate::family::mem::host_store_budget(crate::family::mem::process_resident_bytes());
        let executor = SerialDecode::new(card, eos, sampler_vocab, rows).with_reuse(
            PREFILL_ROWS,
            budget,
            crate::family::step::keep::admit,
        );
        Ok((executor, DecodeSession::default()))
    })?;
    let context = match resolved.recv() {
        Ok(Ok(context)) => context,
        Ok(Err(cause)) => {
            let _ = engine.shutdown();
            return Err(EngError::Checkpoint(format!(
                "the card start failed before the model was ready: {cause}"
            )));
        }
        Err(_) => {
            let report = engine.shutdown();
            return Err(EngError::Checkpoint(format!(
                "the card start failed before the model was ready: {}",
                report
                    .failure
                    .unwrap_or_else(|| "the engine thread ended without a cause".into())
            )));
        }
    };
    Ok((engine, context))
}

/// The model side of one session: the card step over the leaked load, the file's embedding
/// table (it feeds the rows from the host) and the number of rows the card has consumed.
pub(crate) struct CardStep {
    step: LoopStep<'static>,
    embd: GgufMatrix,
    projector: GgufProjector,
    pos: usize,
    /// Counts the sequences that started at position 0 and the clears after a failure.
    generation: u64,
    /// A step failed since the last clear: what the planes hold is unknown.
    failed: bool,
}

impl CardStep {
    pub(crate) fn new(
        step: LoopStep<'static>,
        file: &Arc<GgufWeights>,
        model: &Model,
    ) -> Result<Self, EngError> {
        let (vocab, hidden) = (model.shape.vocab, model.shape.hidden);
        let mut step = step;
        // With the table on the device a greedy step picks its token there and the host reads
        // eight bytes; without it the host gathers the row and reads the logits.
        if let Some(table) =
            device_table(step.context(), file, "token_embd.weight", (vocab, hidden))?
        {
            step.set_embedding(table)?;
        }
        Ok(Self {
            step,
            embd: GgufMatrix::new(file.header(), "token_embd.weight", vocab, hidden)?,
            projector: GgufProjector::new(Arc::clone(file)),
            pos: 0,
            generation: 0,
            failed: false,
        })
    }

    /// Before a step at the stepper's position: a step at position 0 starts a sequence that
    /// writes over the rows of the one before.
    fn begin(&mut self) {
        if self.pos == 0 {
            self.generation += 1;
        }
    }

    /// Notes a failed step, so the next reset clears everything the step carries.
    fn checked<T>(&mut self, result: Result<T, EngError>) -> Result<T, EngError> {
        self.failed |= result.is_err();
        result
    }

    /// The logits row the last step left on the device.
    #[cfg(test)]
    pub(crate) fn last_logits(&self) -> Result<Vec<u8>, EngError> {
        let logits = self.step.logits_row();
        let mut bytes = vec![0u8; logits.bytes()];
        logits.read_at(self.step.stream(), 0, &mut bytes)?;
        Ok(bytes)
    }
}

impl CardStep {
    /// Whether a greedy token of this stepper is picked on the device.
    #[cfg(test)]
    pub(crate) fn picks_on_device(&self) -> bool {
        self.step.takes_tokens()
    }
}

impl TokenStepper for CardStep {
    const OPERATION: &'static str = "lineage_logits";

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        self.feed_rows(&[token])
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
        let rows = self.projector.rows(&self.embd, tokens)?;
        self.begin();
        let fed = self.step.prefill(&rows, self.pos);
        self.checked(fed)?;
        self.pos += tokens.len();
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        let row = self.projector.rows(&self.embd, &[token])?;
        self.begin();
        let stepped = self.step.decode(&row, self.pos);
        self.checked(stepped)?;
        self.pos += 1;
        // The read is enqueued on the step's stream, behind its kernels.
        let logits = self.step.logits_row();
        let mut bytes = vec![0u8; logits.bytes()];
        let read = logits.read_at(self.step.stream(), 0, &mut bytes);
        self.checked(read)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect())
    }

    /// A greedy pick is made on the device when the step holds the embedding table: the token
    /// id goes in, the id of the largest logit comes out, and no row crosses the bus. Any other
    /// sampling, and a request that captures its candidates, takes the logits row to the host;
    /// so does the first token of a request, whose margin over the runner-up the sampler keeps
    /// for the request's report.
    fn next_token(
        &mut self,
        token: u32,
        sampler: &mut Sampler,
        sampler_vocab: usize,
    ) -> Result<u32, EngError> {
        let on_device = self.step.takes_tokens()
            && sampler.is_greedy()
            && !sampler.capture_armed()
            && sampler.first_top2_margin().is_some();
        if !on_device {
            let logits = self.logits(token)?;
            let drawn = sample_logits(Self::OPERATION, &logits, sampler, sampler_vocab);
            return self.checked(drawn);
        }
        self.begin();
        let picked = self.step.decode_token(token, self.pos, sampler_vocab);
        let picked = self.checked(picked)?;
        self.pos += 1;
        sampler.record_device_greedy(picked);
        Ok(picked)
    }

    fn fed(&self) -> usize {
        self.pos
    }

    /// A new sequence: the running state returns to zeros and the cache rows stay, for a later
    /// request that returns to a kept state. After a failed step everything is cleared and no
    /// kept state matches any more.
    fn reset(&mut self) -> Result<(), EngError> {
        if self.failed {
            self.step.reset()?;
            self.generation += 1;
            self.failed = false;
        } else {
            self.step.reset_running()?;
        }
        self.pos = 0;
        Ok(())
    }

    fn kept_bytes(&self) -> Option<u64> {
        Some(self.step.kept_bytes())
    }

    fn keep(&mut self) -> Result<Kept, EngError> {
        let kept = self.step.keep(self.pos);
        // Host-memory denial during a D2H snapshot leaves device state untouched.
        if !matches!(kept, Err(EngError::MemoryPlan(_))) {
            return self.checked(kept);
        }
        kept
    }

    fn put_back(&mut self, kept: &Kept) -> Result<(), EngError> {
        let put = self.step.put_back(kept);
        if !matches!(put, Err(EngError::MemoryPlan(_))) {
            self.checked(put)?;
        } else {
            put?;
        }
        self.pos = kept.rows;
        Ok(())
    }

    fn kept_generation(&self) -> u64 {
        self.generation
    }
}

static CARD_ENGINE_BUILT: AtomicBool = AtomicBool::new(false);

/// Takes the one-engine flag; released on drop unless committed (a failure before the first
/// leak owns nothing on the card).
#[derive(Debug)]
struct BuildGuard {
    committed: bool,
}

impl BuildGuard {
    fn acquire() -> Result<Self, EngError> {
        CARD_ENGINE_BUILT
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                EngError::BadRequest(
                    "engine config: the card step of this lineage builds one engine per process: its weights are leaked for the process lifetime; configure a single entry served by it and restart the process to reload"
                        .to_string(),
                )
            })?;
        Ok(Self { committed: false })
    }

    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for BuildGuard {
    fn drop(&mut self) {
        if !self.committed {
            CARD_ENGINE_BUILT.store(false, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flag belongs to this test alone: the other tests of the crate do not start the card
    /// engine, and the store puts it back for the rest of the run.
    #[test]
    fn a_committed_guard_keeps_the_flag_and_an_uncommitted_one_releases_it() {
        assert!(!CARD_ENGINE_BUILT.load(Ordering::Acquire));
        let uncommitted = BuildGuard::acquire().expect("the first build takes the flag");
        let refused = BuildGuard::acquire().expect_err("a second build in one process");
        assert!(
            refused.describe().contains("one engine per process"),
            "{refused:?}"
        );
        drop(uncommitted);
        assert!(!CARD_ENGINE_BUILT.load(Ordering::Acquire));

        let mut committed = BuildGuard::acquire().expect("the flag was released");
        committed.commit();
        drop(committed);
        assert!(CARD_ENGINE_BUILT.load(Ordering::Acquire));
        CARD_ENGINE_BUILT.store(false, Ordering::Release);
    }
}
