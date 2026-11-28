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
use crate::family::session::{DecodeSession, Kept};
use crate::family::step::stepper::{DeviceStep, DeviceStepper};
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

/// The model side of one session: the card step over the leaked load and the file's embedding
/// table (it feeds the rows from the host when no device table is resident).
pub(crate) struct CardDevice {
    step: LoopStep<'static>,
    embd: GgufMatrix,
    projector: GgufProjector,
}

/// The served stepper of the Qwen lineage: shared position/generation/failure tracking over
/// [`CardDevice`].
pub(crate) type CardStep = DeviceStepper<CardDevice>;

impl CardStep {
    pub(crate) fn new(
        mut step: LoopStep<'static>,
        file: &Arc<GgufWeights>,
        model: &Model,
    ) -> Result<Self, EngError> {
        let (vocab, hidden) = (model.shape.vocab, model.shape.hidden);
        // With the table on the device a greedy step picks its token there and the host reads
        // eight bytes; without it the host gathers the row and reads the logits.
        if let Some(table) =
            device_table(step.context(), file, "token_embd.weight", (vocab, hidden))?
        {
            step.set_embedding(table)?;
        }
        Ok(DeviceStepper::over(CardDevice {
            step,
            embd: GgufMatrix::new(file.header(), "token_embd.weight", vocab, hidden)?,
            projector: GgufProjector::new(Arc::clone(file)),
        }))
    }

    /// The logits row the last step left on the device.
    #[cfg(test)]
    pub(crate) fn last_logits(&self) -> Result<Vec<u8>, EngError> {
        let logits = self.device().step.logits_row();
        let mut bytes = vec![0u8; logits.bytes()];
        logits.read_at(self.device().step.stream(), 0, &mut bytes)?;
        Ok(bytes)
    }

    /// Whether a greedy token of this stepper is picked on the device.
    #[cfg(test)]
    pub(crate) fn picks_on_device(&self) -> bool {
        self.device().step.takes_tokens()
    }
}

impl DeviceStep for CardDevice {
    const OPERATION: &'static str = "lineage_logits";

    fn feed_rows(&mut self, tokens: &[u32], pos: usize) -> Result<(), EngError> {
        let rows = self.projector.rows(&self.embd, tokens)?;
        self.step.prefill(&rows, pos)
    }

    fn logits(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, EngError> {
        let row = self.projector.rows(&self.embd, &[token])?;
        self.step.decode(&row, pos)?;
        // The read is enqueued on the step's stream, behind its kernels.
        let logits = self.step.logits_row();
        let mut bytes = vec![0u8; logits.bytes()];
        logits.read_at(self.step.stream(), 0, &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect())
    }

    fn takes_tokens(&self) -> bool {
        self.step.takes_tokens()
    }

    fn decode_token(
        &mut self,
        token: u32,
        pos: usize,
        sampler_vocab: usize,
    ) -> Result<u32, EngError> {
        self.step.decode_token(token, pos, sampler_vocab)
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.step.reset()
    }

    fn reset_running(&mut self) -> Result<(), EngError> {
        self.step.reset_running()
    }

    fn kept_bytes(&self) -> u64 {
        self.step.kept_bytes()
    }

    fn keep(&self, rows: usize) -> Result<Kept, EngError> {
        self.step.keep(rows)
    }

    fn put_back(&mut self, kept: &Kept) -> Result<(), EngError> {
        self.step.put_back(kept)
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
