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
//! `ResidentDecoder::new_with_kv`, and the cache form (packet KV8-INTEGRATE) with them:
//! `kv_dtype` `f32` is the control and `q8` keeps the 8-bit codes. The chunk path is dtype aware
//! (packet Q36-BF16-b), so a BF16 engine plans its chunk exactly like an F32 engine when the
//! budget allows one.

use super::resident::executor::{Q36Executor, Q36Rules};
use super::settings::{NO_BUDGET_PIECE, Q36Settings, tokenizer_pin};
use crate::{
    EngError,
    family::session::StoreLimits,
    sched::worker::{Engine, spawn_engine},
    tokenizer::TokenizerBundle,
};

/// The prefix-store limits of a `prefix_reuse` engine (decision 0036, addendum 2): two pinned
/// roots and six LRU entries. An engine without the flag keeps the addendum 1 store.
const PREFIX_STORE_LIMITS: StoreLimits = StoreLimits {
    roots: 2,
    entries: 6,
};

/// The rules of one engine, built from the settings and the pinned tokenizer facts. Every
/// setting that the decoder reads travels here in one place: `compile` is the `qwen36.compile`
/// member, so a settings change reaches the decoder exactly through this function, and
/// `prefix_reuse` (set by the factory after `Q36Settings::new`) picks the prefix-store limits:
/// the addendum 2 set with the flag on, the addendum 1 store without it.
fn rules_of(settings: &Q36Settings, eos: Vec<u32>, sampler_vocab: usize) -> Q36Rules {
    Q36Rules {
        eos,
        sampler_vocab,
        context_limit: settings.context_limit,
        session_cap: settings.session_cap,
        activation: settings.activation,
        kv_dtype: settings.kv,
        compile: settings.compile,
        prefix_store: if settings.prefix_reuse {
            PREFIX_STORE_LIMITS
        } else {
            StoreLimits::default()
        },
    }
}

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
    let rules = rules_of(settings, eos, sampler_vocab);
    let engine = spawn_engine(fallback, move || {
        let (mut executor, session) = Q36Executor::open(&owned.artifact, owned.cache_limit, rules)?;
        let (delta_warmup, moe_warmup) = executor.compile_warmup_seconds();
        if !delta_warmup.is_empty() || !moe_warmup.is_empty() {
            let total = |seconds: &[f64]| seconds.iter().sum::<f64>();
            eprintln!(
                "q36 compile: warm-up traced {} delta program(s) in {:.3} s and {} moe program(s) \
                 in {:.3} s",
                delta_warmup.len(),
                total(delta_warmup),
                moe_warmup.len(),
                total(moe_warmup)
            );
        }
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

#[cfg(test)]
mod tests {
    use super::rules_of;
    use crate::family::qwen36::settings::{Backend, Q36Env, Q36Options, Q36Settings};
    use crate::family::session::StoreLimits;
    use std::path::Path;

    /// The rules of the engine thread carry the `qwen36.compile` member of the settings: `start`
    /// builds its rules through `rules_of`, so a config entry with `compile: true` reaches the
    /// decoder and an entry without it does not. Mutation check (the start-passes-false patch):
    /// with the member dropped here, the `rules.compile` row reads false and the test fails.
    #[test]
    fn the_rules_carry_the_compile_setting_of_the_config() {
        let settings = |options: Option<&Q36Options>| {
            Q36Settings::new(
                Path::new("/m"),
                Path::new("/t"),
                options,
                &Q36Env::default(),
                Backend::Mlx,
            )
        };
        // The MLX arm requires a cache limit: the member stands in for `Q36_CACHE_LIMIT`.
        let with = settings(Some(&Q36Options {
            cache_limit_bytes: Some(1024),
            compile: Some(true),
            ..Q36Options::default()
        }))
        .unwrap();
        assert!(with.compile, "the MLX arm reads the compile member");
        assert!(
            rules_of(&with, vec![248_046], 8).compile,
            "the rules carry the compile setting into the executor"
        );
        let without = settings(Some(&Q36Options {
            cache_limit_bytes: Some(1024),
            ..Q36Options::default()
        }))
        .unwrap();
        assert!(!without.compile, "the default is off");
        assert!(
            !rules_of(&without, vec![248_046], 8).compile,
            "off stays off through the whole start path"
        );
    }

    /// The rules carry the prefix-store limits of the `prefix_reuse` flag (packet SERVE-2-c
    /// item 1; decision 0036, addendum 2): the factory sets the flag on the settings after
    /// `new`, and `rules_of` maps it to the addendum 2 limits (two pinned roots, six LRU
    /// entries); with the flag off the rules keep the addendum 1 store (zero roots, one
    /// entry), so every engine started without the flag behaves exactly as before.
    ///
    /// Mutation check (the limits-ignored patch): with `StoreLimits::default()` in both arms,
    /// the (2, 6) row fails.
    #[test]
    fn the_rules_carry_the_prefix_store_limits_of_the_prefix_reuse_flag() {
        let settings = |flag: bool| {
            let mut settings = Q36Settings::new(
                Path::new("/m"),
                Path::new("/t"),
                Some(&Q36Options {
                    cache_limit_bytes: Some(1024),
                    ..Q36Options::default()
                }),
                &Q36Env::default(),
                Backend::Mlx,
            )
            .unwrap();
            // The factory sets the flag after `new`; the test does the same.
            settings.prefix_reuse = flag;
            settings
        };
        assert_eq!(
            rules_of(&settings(true), vec![248_046], 8).prefix_store,
            StoreLimits {
                roots: 2,
                entries: 6
            },
            "a prefix_reuse engine keeps the addendum 2 set (a literal, not the constant)"
        );
        assert_eq!(
            rules_of(&settings(false), vec![248_046], 8).prefix_store,
            StoreLimits {
                roots: 0,
                entries: 1
            },
            "a literal, not the default under test: off keeps the addendum 1 store"
        );
    }
}
