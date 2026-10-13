//! Starts the Q36 engine of one `enginesrv` config entry (packet Q36-F3b).
//!
//! The tokenizer bundle loads on the calling thread. It carries only portable data. The engine
//! thread builds everything native inside the `make` closure (`Q36Executor::open`), so no native
//! value crosses a thread boundary. The tokenizer files come from `tokenizer_dir` and are checked
//! against the pinned digests of `settings::tokenizer_pin`, so another template or tokenizer
//! fails at load. The EOS ids (248046 and 248044) come from the pinned `generation_config.json`.
//!
//! Chunked prefill (decision 0044; packet Q36-E10): the chunk path stays off unless the settings
//! carry `prefill_budget_bytes`. Then the engine thread calls `Q36Executor::plan_prefill` once,
//! after the weights are resident and before the first job, with `prefill_chunk` as the largest
//! chunk. The plan result decides the worker piece: the executor reports the planned chunk through
//! `ExecutorOps::prefill_piece`, so a planned chunk is never larger than a piece. Without a plan,
//! or with a plan that admits no chunk, the piece is 16 (`NO_BUDGET_PIECE`), the fallback that
//! `spawn_engine` receives. A plan that admits no chunk is not an error: the engine runs
//! one-token steps and the start-up line says why.
//!
//! The activation dtype (packet Q36-BF16-a) travels with the rules into
//! `ResidentDecoder::new_with_activation`. The chunk path is dtype aware (packet Q36-BF16-b), so
//! a BF16 engine plans its chunk exactly like an F32 engine when the budget allows one.

use super::resident::executor::{Q36Executor, Q36Rules};
use super::settings::{NO_BUDGET_PIECE, Q36Settings, tokenizer_pin};
use crate::{
    EngError,
    sched::worker::{Engine, spawn_engine},
    tokenizer::TokenizerBundle,
};

/// Loads and pins the tokenizer, then starts the engine thread. The thread loads the weights, so
/// the engine state stays `Loading` until the weights are resident. Returns the engine and the
/// pinned tokenizer bundle that renders and encodes its prompts.
pub(crate) fn start(settings: &Q36Settings) -> Result<(Engine, TokenizerBundle), EngError> {
    if settings.capture {
        // `capture` is a CUDA key; one config schema serves both arms, so the MLX arm reads
        // it, ignores it, and says so on start-up (CUDA-SERVE-1 2.3).
        eprintln!("q36: capture is a CUDA key and is ignored on this arm");
    }
    let bundle = TokenizerBundle::load(&settings.tokenizer_dir, &tokenizer_pin()?)?;
    let eos = bundle.eos_token_ids().to_vec();
    let sampler_vocab = usize::try_from(bundle.vocab_size())
        .map_err(|_| EngError::Checkpoint("tokenizer vocabulary exceeds usize".to_string()))?;
    let fallback = usize::try_from(NO_BUDGET_PIECE).map_err(|_| {
        EngError::BadRequest("engine config: the worker piece exceeds usize".into())
    })?;
    let owned = settings.clone();
    let rules = Q36Rules {
        eos,
        sampler_vocab,
        context_limit: settings.context_limit,
        session_cap: settings.session_cap,
        activation: settings.activation,
    };
    let engine = spawn_engine(fallback, move || {
        let (mut executor, session) = Q36Executor::open(&owned.artifact, owned.cache_limit, rules)?;
        if let Some(line) = plan_prefill(&mut executor, &owned)? {
            eprintln!("{line}");
        }
        Ok((executor, session))
    })?;
    Ok((engine, bundle))
}

/// Plans the chunked prefill of `executor` from the settings and returns the line to log. Without
/// `prefill_budget` it does nothing and returns `None`, and the executor stays on one-token steps.
/// With a budget, a plan that admits no chunk also succeeds: the line names the terms, and the
/// executor stays on one-token steps. Only a failed plan returns an error. Both activation dtypes
/// plan a chunk (packet Q36-BF16-b): the chunk path is dtype aware, and only the budget or the
/// largest chunk can keep a chunk out.
pub(super) fn plan_prefill(
    executor: &mut Q36Executor,
    settings: &Q36Settings,
) -> Result<Option<String>, EngError> {
    let Some(budget) = settings.prefill_budget else {
        return Ok(None);
    };
    let max_rows = settings.prefill_chunk;
    let plan = executor.plan_prefill(budget, settings.prefill_reserve, max_rows)?;
    let terms = plan.terms.to_string().lines().collect::<Vec<_>>().join(" ");
    Ok(Some(match plan.chunk {
        Some(rows) => {
            format!("q36 prefill: chunk of {rows} rows planned (largest {max_rows}); {terms}")
        }
        None if max_rows == 1 => format!(
            "q36 prefill: the largest chunk is 1 (prefill_chunk); prompts run one token at a time; {terms}"
        ),
        None => format!(
            "q36 prefill: no chunk above one token fits (largest {max_rows}); prompts run one token at a time; {terms}"
        ),
    }))
}
