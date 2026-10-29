//! Starts the Q36 engine of one `enginesrv` config entry.
//!
//! The tokenizer bundle loads on the calling thread. It carries only portable data. The engine
//! thread builds everything native inside the `make` closure (`Q36Executor::open`), so no native
//! value crosses a thread boundary. The tokenizer files come from `tokenizer_dir` and are checked
//! against the pinned digests of `settings::tokenizer_pin`, so another template or tokenizer
//! fails at load. The EOS ids (248046 and 248044) come from the pinned `generation_config.json`.
//!
//! Chunked prefill: the chunk path stays off unless the settings carry `prefill_budget_bytes`. Then
//! the engine thread calls `Q36Executor::plan_prefill` once, after the weights are resident and
//! before the first job, with `prefill_chunk` as the largest chunk. The plan result decides the
//! worker piece: the executor reports the planned chunk through `ExecutorOps::prefill_piece`, so a
//! planned chunk is never larger than a piece. Without a plan, or with a plan that admits no chunk,
//! the piece is 16 (`NO_BUDGET_PIECE`), the fallback that `spawn_engine` receives. A plan that
//! admits no chunk is not an error: the engine runs one-token steps and the start-up line says why.
//!
//! The activation dtype travels with the rules into `ResidentDecoder::new_with_kv`, and the cache
//! form with them: `kv_dtype` `f32` is the control and `q8` keeps the 8-bit codes. The chunk path
//! is dtype aware, so a BF16 engine plans its chunk exactly like an F32 engine when the budget
//! allows one.

use super::resident::executor::{Q36Executor, Q36Rules};
use super::settings::{NO_BUDGET_PIECE, Q36Settings, tokenizer_pin};
use crate::{
    EngError,
    sched::worker::{Engine, ReadyNote, spawn_engine_noted},
    tokenizer::TokenizerBundle,
};

/// The rules of one engine, built from the settings and the pinned tokenizer facts. Every setting
/// that the decoder reads travels here in one place: `compile` is the `qwen36.compile` member, so a
/// settings change reaches the decoder exactly through this function, and `prefix_reuse` (set by
/// the factory after `Q36Settings::new`) picks the prefix-store limits: two pinned roots and six
/// LRU entries with the flag on, the one-entry store without it.
fn rules_of(settings: &Q36Settings, eos: Vec<u32>, sampler_vocab: usize) -> Q36Rules {
    Q36Rules {
        eos,
        sampler_vocab,
        context_limit: settings.context_limit,
        session_cap: settings.session_cap,
        activation: settings.activation,
        kv_dtype: settings.kv,
        compile: settings.compile,
        max_seqs: settings.max_seqs,
        // The default stays six entries for every spec (an entry's real cost grows with its depth,
        // so no budget-derived default ships before the replay measurement); the exact
        // `prefix_store_entries` override reaches every spec through here.
        prefix_store: super::settings::prefix_store_limits_of(
            settings.prefix_reuse,
            settings.prefix_store_entries,
        ),
    }
}

/// Loads and pins the tokenizer, then starts the engine thread. The thread loads the weights, so
/// the engine state stays `Loading` until the weights are resident. Returns the engine and the
/// pinned tokenizer bundle that renders and encodes its prompts.
pub(crate) fn start(
    settings: &Q36Settings,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle), EngError> {
    if settings.capture {
        // `capture` is a CUDA key; one config schema serves both arms, so the MLX arm reads
        // it, ignores it, and says so on start-up.
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
    // The eviction-cliff boot report, once per engine: the limits the rules carry are the
    // ones the executor builds the store with, so this is the effective shape at ready
    // time. The budget an entry charges is the SESSION CAP (`SessionBudget::new(rules.
    // session_cap)`), not the MLX allocator cache limit, so the line labels it that way;
    // a set member with the flag off is ignored, and says so.
    if settings.prefix_reuse {
        eprintln!(
            "q36: prefix store (roots {}, entries {}, session cap {} MiB)",
            rules.prefix_store.roots,
            rules.prefix_store.entries,
            settings.session_cap / (1024 * 1024),
        );
    } else if settings.prefix_store_entries.is_some() {
        eprintln!(
            "q36: prefix_store_entries is set but prefix_reuse is off; the member is ignored"
        );
    }
    let engine = spawn_engine_noted(fallback, ready_note, move || {
        let (mut executor, session) =
            Q36Executor::open_for_family(&owned.artifact, owned.cache_limit, rules, &owned.spec)?;
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
/// With a budget, a single-session plan that admits no chunk also succeeds. Concurrent serving
/// requires an admitted chunk; it never falls back to single-token prefill. Both activation dtypes
/// plan a chunk: the chunk path is dtype aware, and only the budget or the largest chunk can keep a
/// chunk out.
pub(super) fn plan_prefill(
    executor: &mut Q36Executor,
    settings: &Q36Settings,
) -> Result<Option<String>, EngError> {
    let Some(budget) = settings.prefill_budget else {
        if settings.max_seqs > 1 {
            return Err(EngError::BadRequest(
                "concurrent serving requires a prefill budget".into(),
            ));
        }
        return Ok(None);
    };
    let max_rows = settings.prefill_chunk;
    // Concurrent live states can fill the aggregate cap, not just one context's reserve.
    let reserve = if settings.max_seqs > 1 {
        settings.prefill_reserve.max(settings.session_cap)
    } else {
        settings.prefill_reserve
    };
    let plan = executor.plan_prefill(budget, reserve, max_rows)?;
    if settings.max_seqs > 1 && plan.chunk.is_none() {
        return Err(EngError::MemoryPlan(
            "concurrent serving requires an admitted prefill chunk above one token".into(),
        ));
    }
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
    use crate::family::{FamilySpec, QWEN3_6_35B_A3B, QWEN3_8_27B_DENSE};
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
            max_seqs: Some(4),
            prefill_budget_bytes: Some(8 << 30),
            prefill_session_reserve_bytes: Some(0),
            ..Q36Options::default()
        }))
        .unwrap();
        assert!(with.compile, "the MLX arm reads the compile member");
        assert_eq!(rules_of(&with, vec![248_046], 8).max_seqs, 4);
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
        assert_eq!(rules_of(&without, vec![248_046], 8).max_seqs, 1);
        assert!(
            !rules_of(&without, vec![248_046], 8).compile,
            "off stays off through the whole start path"
        );
    }

    #[test]
    fn concurrent_prefill_reserves_the_aggregate_live_state_cap() {
        use crate::family::qwen36::{
            host_load::HostWeights,
            resident::executor::Q36Executor,
            tiny::{self, TINY_GQA},
        };
        use crate::kernels::mlx::MlxKernels;
        let _serial = crate::native_test_lock();
        let mut settings = Q36Settings::new(
            Path::new("/m"),
            Path::new("/t"),
            Some(&Q36Options {
                cache_limit_bytes: Some(1024),
                context_limit: Some(8),
                kv_dtype: Some("f32".into()),
                compile: Some(true),
                max_seqs: Some(1),
                prefill_budget_bytes: Some(3 << 30),
                prefill_chunk: Some(4),
                prefill_session_reserve_bytes: Some(0),
                ..Q36Options::default()
            }),
            &Q36Env::default(),
            Backend::Mlx,
        )
        .unwrap();
        let ckpt = Box::leak(Box::new(tiny::checkpoint(&TINY_GQA)));
        let host = HostWeights::load(ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap();
        let (mut executor, _session) = Q36Executor::from_parts(
            MlxKernels::try_new().unwrap(),
            ckpt,
            host,
            rules_of(&settings, vec![], 8),
        )
        .unwrap();
        settings.max_seqs = 1;
        super::plan_prefill(&mut executor, &settings).unwrap();
        assert!(
            executor.prefill_chunk().is_some(),
            "the solo control must admit a chunk"
        );
        settings.max_seqs = 4;
        assert!(matches!(
            super::plan_prefill(&mut executor, &settings),
            Err(crate::EngError::MemoryPlan(_))
        ));
        assert_eq!(
            executor.prefill_chunk(),
            None,
            "the aggregate cap exceeds this budget"
        );
    }

    /// The rules carry the prefix-store limits of the `prefix_reuse` flag: the factory sets the
    /// flag on the settings after `new`, and `rules_of` maps it to two pinned roots and six LRU
    /// entries; with the flag off the rules keep the one-entry store (zero roots, one entry), so
    /// every engine started without the flag behaves exactly as before.
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
            "a prefix_reuse engine keeps two roots and six entries (a literal, not the constant)"
        );
        assert_eq!(
            rules_of(&settings(false), vec![248_046], 8).prefix_store,
            StoreLimits {
                roots: 0,
                entries: 1
            },
            "a literal, not the default under test: off keeps the one-entry store"
        );
    }

    /// The DEFAULT stays six entries through `rules_of` for every spec and whatever the budget is,
    /// and the exact `prefix_store_entries` override reaches every spec - dense Qwen3.8 included -
    /// with a valid explicit session cap (dense admission refuses the fixture without one).
    #[test]
    fn the_rules_carry_the_default_six_and_the_exact_override_for_every_spec() {
        let built = |spec: FamilySpec, set: Option<usize>| {
            let mut settings = Q36Settings::new_for_family(
                Path::new("/m"),
                Path::new("/t"),
                Some(&Q36Options {
                    cache_limit_bytes: Some(4 * 1024 * 1024 * 1024),
                    // Dense admission requires an explicit cap covering its configured
                    // context; the value is the F32 dense requirement at 32,768 positions.
                    session_cap_bytes: Some(2_750_218_240),
                    context_limit: Some(32_768),
                    prefix_store_entries: set,
                    ..Q36Options::default()
                }),
                &Q36Env::default(),
                Backend::Mlx,
                spec,
            )
            .unwrap();
            settings.prefix_reuse = true;
            rules_of(&settings, vec![248_046], 8).prefix_store
        };
        assert_eq!(
            built(QWEN3_6_35B_A3B, None),
            StoreLimits {
                roots: 2,
                entries: 6
            },
            "the default stays six through the rules, whatever the budget"
        );
        assert_eq!(
            built(QWEN3_6_35B_A3B, Some(32)),
            StoreLimits {
                roots: 2,
                entries: 32
            },
            "the exact override wins through the rules"
        );
        assert_eq!(
            built(QWEN3_8_27B_DENSE, None),
            StoreLimits {
                roots: 2,
                entries: 6
            },
            "dense keeps six entries too: the default changed for no spec"
        );
        assert_eq!(
            built(QWEN3_8_27B_DENSE, Some(32)),
            StoreLimits {
                roots: 2,
                entries: 32
            },
            "the override reaches dense as well"
        );
    }
}
