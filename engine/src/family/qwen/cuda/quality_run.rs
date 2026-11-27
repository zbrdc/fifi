//! The teacher-forced quality leg of the lineage's step: a model of the lineage scored over a
//! fixed id stream through [`crate::family::teacher_forced`], so `scripts/kld_vs_llamacpp` pairs
//! its sidecars with another engine's or with its own exact route.
//!
//! The leg prefills each window's unscored tokens through the chunked prefill, then feeds one
//! forced token per scored position through the decode step. With `KVQ_EXACT` set the products
//! decode their weights over f32 rows and the prompt's attention scores in f32: that run is the
//! reference the other arms are scored against. `KVQ_EXACT=chunks` or `=rows` does so for one
//! of the two routes only.

use std::rc::Rc;
use std::sync::Arc;

use super::step::LoopStep;
use crate::EngError;
use crate::family::project::Projector;
use crate::family::project::gguf::{GgufMatrix, GgufProjector};
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
#[ignore = "a card under fifi-gpu-run: KVQ_MODEL, KVQ_SAMPLER_VOCAB, KVQ_IDS, KVQ_IDS_SHA, KVQ_LAYOUT, KVQ_WINDOWS, KVQ_OUT, optional KVQ_REF, KVQ_EXACT, KVQ_EXACT_CLASS and KVQ_FEED_ROWS"]
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
    let (model, routing) = super::serve::model_of(&file).expect("a model of the lineage");
    let ctx = Rc::new(CudaContext::new().expect("context"));
    let loaded = super::serve::load_model(&ctx, &file, &model).expect("load");
    let mut step =
        LoopStep::open_model(ctx.clone(), &loaded, &model, routing, run.context + 1).expect("step");
    // `KVQ_EXACT=chunks` puts the prompt pieces of two rows or more on the decoding routes,
    // `=rows` the one-row steps (the scored positions, and a prompt's last piece when it is one
    // row); any other value puts both.
    let exact = std::env::var("KVQ_EXACT").ok();
    let (rows, chunks) = match exact.as_deref() {
        None => (false, false),
        Some("chunks") => (false, true),
        Some("rows") => (true, false),
        Some(_) => (true, true),
    };
    step.set_exact_routes(rows, chunks);
    println!("kvq lineage leg: {} exact={exact:?}", model.family);
    let shape = model.shape;
    // `KVQ_EXACT_CLASS` puts one class of products of the looped Q4_K_M file on the decoding
    // routes in every step and chunk, the others staying on the served ones: the distance from
    // the reference that goes away is at least that class's share. A class is a (rows, inputs,
    // type) triple, so it holds only the layers that store the tensor in the named type (the
    // file mixes types per layer), and `mixer-out` holds the delta rule's and the attention's
    // output projections alike, which have one shape.
    if let Ok(name) = std::env::var("KVQ_EXACT_CLASS") {
        use crate::family::step::rows::exact_class;
        use crate::kernels::cuda::resident::matvec_q::QuantType::{Q4K, Q6K};
        let (h, ff) = (shape.hidden, shape.feed_forward);
        let (q, kv) = (
            shape.heads_q * 2 * shape.head_dim,
            shape.heads_kv * shape.head_dim,
        );
        let class = match name.as_str() {
            "delta-inputs" => (shape.qkvz_dim(), h, Q6K),
            "delta-logits" => (2 * shape.delta_value_heads, h, Q4K),
            "mixer-out" => (h, shape.delta_inner(), Q4K),
            "ffn-gate-up" => (ff, h, Q4K),
            "ffn-down" => (h, ff, Q6K),
            "attn-q" => (q, h, Q4K),
            "attn-k" => (kv, h, Q4K),
            "attn-v" => (kv, h, Q6K),
            other => panic!("KVQ_EXACT_CLASS: no class named {other}"),
        };
        exact_class::set(Some(class));
        println!("kvq lineage leg: class {name} = {class:?} on the decoding routes");
    }
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
