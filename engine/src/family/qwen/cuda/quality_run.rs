//! The teacher-forced quality leg of the lineage's step: a model of the lineage scored over a
//! fixed id stream through [`crate::family::teacher_forced`], so `scripts/kld_vs_llamacpp` pairs
//! its sidecars with another engine's or with its own exact route.
//!
//! The leg prefills each window's unscored tokens through the chunked prefill, then feeds one
//! forced token per scored position through the decode step. With `KVQ_EXACT` set every product
//! decodes its weights over f32 rows: that run is the reference the other arms are scored against.

use std::rc::Rc;
use std::sync::Arc;

use super::step::{LoopStep, Routing, looped};
use crate::EngError;
use crate::family::project::Projector;
use crate::family::project::cuda::{BorrowedLoader, LoadOptions, WeightLayout};
use crate::family::project::gguf::{GgufMatrix, GgufProjector};
use crate::family::qwen::recipe::{layers, routed_experts, tensors};
use crate::family::qwen::sensenova::weights::LoopWeights;
use crate::family::session::TokenStepper;
use crate::family::teacher_forced::{Run, score_windows};
use crate::kernels::cuda::CudaContext;
use crate::weights::{GgufLimits, GgufWeights};

/// Prompt rows fed per call; the step cuts them into its own chunks.
const FEED_ROWS: usize = 1024;

/// The step as a token stepper: embedding rows gathered on the host, the logits row read back.
struct Stepper<'a> {
    step: LoopStep<'a>,
    projector: GgufProjector,
    embd: GgufMatrix,
    pos: usize,
}

impl TokenStepper for Stepper<'_> {
    const OPERATION: &'static str = "lineage_quality_logits";

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        self.feed_rows(&[token])
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
        let rows = self.projector.rows(&self.embd, tokens)?;
        self.step.prefill(&rows, self.pos)?;
        self.pos += tokens.len();
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        let row = self.projector.rows(&self.embd, &[token])?;
        self.step.decode(&row, self.pos)?;
        self.pos += 1;
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

/// Scores the windows `KVQ_WINDOWS` of the lineage's model `KVQ_MODEL` on the lineage's step.
#[test]
#[ignore = "a card under fifi-gpu-run: KVQ_MODEL, KVQ_SAMPLER_VOCAB, KVQ_IDS, KVQ_IDS_SHA, KVQ_LAYOUT, KVQ_WINDOWS, KVQ_OUT, optional KVQ_REF, KVQ_EXACT and KVQ_FEED_ROWS"]
fn lineage_quality_leg() {
    let env = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"));
    let run = Run::from_env().expect("run");
    let sampler_vocab: usize = env("KVQ_SAMPLER_VOCAB").parse().expect("KVQ_SAMPLER_VOCAB");
    let file = Arc::new(
        GgufWeights::open_any_type(
            &std::path::PathBuf::from(env("KVQ_MODEL")),
            GgufLimits {
                max_header_bytes: 32 << 20,
                max_tensor_bytes: 8 << 30,
            },
            None,
        )
        .expect("model"),
    );
    // The routed-expert model has one residual stream: the routing of a two-stream range is
    // unused there. Any other header is read as the looped model.
    let (model, routing) = match routed_experts(file.header()) {
        Ok(model) => (
            model,
            Routing {
                pre_scale: 1.0,
                iterations: 1,
                fixed_identity: true,
            },
        ),
        Err(_) => {
            let weights = LoopWeights::open(file.clone()).expect("looped model");
            (
                looped(weights.shape(), weights.plan(), weights.rope()),
                weights.plan().routing(),
            )
        }
    };
    let names = tensors(&layers(&model.visits, model.blocks).0);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let ctx = Rc::new(CudaContext::new().expect("context"));
    let loaded = BorrowedLoader::new(&ctx)
        .expect("loader")
        .load_sliced(
            &file,
            &refs,
            LoadOptions {
                layout: WeightLayout::Native,
                ..LoadOptions::default()
            },
            &[],
            0,
        )
        .expect("load");
    let mut step =
        LoopStep::open_model(ctx.clone(), &loaded, &model, routing, run.context + 1).expect("step");
    let exact = std::env::var_os("KVQ_EXACT").is_some();
    step.set_exact(exact);
    println!("kvq lineage leg: {} exact={exact}", model.family);
    let shape = model.shape;
    let mut stepper = Stepper {
        step,
        projector: GgufProjector::new(file.clone()),
        embd: GgufMatrix::new(
            file.header(),
            "token_embd.weight",
            shape.vocab,
            shape.hidden,
        )
        .expect("embedding"),
        pos: 0,
    };
    // `KVQ_FEED_ROWS` feeds the prompt in smaller pieces. Under the exact switch one row at a
    // time takes the one-row decoding product where a chunk takes the batched one: a second
    // exact arithmetic to hold the first against.
    let feed = match std::env::var("KVQ_FEED_ROWS") {
        Ok(rows) => rows.parse().expect("KVQ_FEED_ROWS"),
        Err(_) => FEED_ROWS,
    };
    score_windows(&mut stepper, &run, feed, sampler_vocab).expect("score");
}
