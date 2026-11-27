//! Serves SenseNova-Loop from the card step behind `FIFI_SENSENOVA_CUDA=1`: a stepper over
//! [`LoopStep`] for the shared serial decode session. It follows `flash_next/cuda/start.rs`.
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

use super::super::serve::LIMITS;
use super::super::weights::LoopWeights;
use crate::EngError;
use crate::family::executor::SerialDecode;
use crate::family::project::Projector;
use crate::family::project::cuda::{
    BorrowedLoader, CudaWeights, LoadOptions, WeightLayout, served_admission,
};
use crate::family::project::gguf::GgufProjector;
use crate::family::qwen::cuda::load;
use crate::family::qwen::cuda::step::{LoopStep, PREFILL_ROWS};
use crate::family::session::{DecodeSession, TokenStepper};
use crate::kernels::cuda::CudaContext;
use crate::sched::worker::{Engine, ReadyNote, spawn_engine_noted};
use crate::weights::GgufWeights;

/// Starts the card engine of one GGUF file. The context limit is also the step's cache depth.
pub(crate) fn start(
    file: PathBuf,
    ready_note: ReadyNote,
    eos: Vec<u32>,
    sampler_vocab: usize,
    context_limit: u32,
) -> Result<Engine, EngError> {
    let mut guard = BuildGuard::acquire()?;
    let cache_rows = context_limit as usize;
    let engine = spawn_engine_noted(PREFILL_ROWS, ready_note, move |_| {
        let began = Instant::now();
        let ctx = Rc::new(CudaContext::with_admission(&served_admission())?);
        let gguf = Arc::new(GgufWeights::open_any_type(&file, LIMITS, None)?);
        let host = LoopWeights::open(Arc::clone(&gguf))?;
        let loaded = load::load(
            &BorrowedLoader::new(&ctx)?,
            &gguf,
            host.plan(),
            LoadOptions {
                layout: WeightLayout::Native,
                ..LoadOptions::default()
            },
        )?;
        drop(gguf);
        // Past this point the load is the process's: see the module note.
        guard.commit();
        let loaded: &'static CudaWeights = Box::leak(Box::new(loaded));
        let step = LoopStep::open(
            Rc::clone(&ctx),
            loaded,
            host.shape(),
            host.plan(),
            host.rope(),
            cache_rows,
        )?;
        eprintln!(
            "sensenova cuda: ready in {:.1} s; the card step serves",
            began.elapsed().as_secs_f64()
        );
        let card = CardStep::new(step, host);
        let executor = SerialDecode::new(card, eos, sampler_vocab, cache_rows);
        Ok((executor, DecodeSession::default()))
    })?;
    Ok(engine)
}

/// The model side of one session: the card step over the leaked load, the host view of the file
/// (its embedding table feeds the rows), and the number of rows the card has consumed.
pub(crate) struct CardStep {
    step: LoopStep<'static>,
    host: LoopWeights,
    projector: GgufProjector,
    pos: usize,
}

impl CardStep {
    pub(crate) fn new(step: LoopStep<'static>, host: LoopWeights) -> Self {
        let projector = host.projector();
        Self {
            step,
            host,
            projector,
            pos: 0,
        }
    }
}

impl TokenStepper for CardStep {
    const OPERATION: &'static str = "sensenova_logits";

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        self.feed_rows(&[token])
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
        let rows = self.projector.rows(self.host.token_embd(), tokens)?;
        self.step.prefill(&rows, self.pos)?;
        self.pos += tokens.len();
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        let row = self.projector.rows(self.host.token_embd(), &[token])?;
        self.step.decode(&row, self.pos)?;
        self.pos += 1;
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

    fn fed(&self) -> usize {
        self.pos
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.step.reset()?;
        self.pos = 0;
        Ok(())
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
                    "engine config: the SenseNova CUDA backend builds one engine per process: its weights are leaked for the process lifetime; configure a single sensenova entry and restart the process to reload"
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
