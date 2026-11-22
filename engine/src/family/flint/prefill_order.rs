// The CUDA executor and the diagnostic surfaces (`check`, `config effective`) are the consumers;
// a default-feature lib build compiles the module for the diagnostics without reaching every rule.
#![cfg_attr(not(any(test, feature = "native-cuda")), allow(dead_code))]

//! Which prefill order a Flint CUDA engine opens with. The decision is pure so the host
//! tests cover it: the open passes the selections of `FIFI_EXL3_LAYER_MAJOR` and
//! `FIFI_EXL3_TALL_HILO` and the facts of the options, and gets back the order, the hi/lo
//! state, where each came from (default or explicit) and the reason it logs.
//!
//! The default for a Flint load is the job-table pass with the hi/lo split of its tall
//! projections, on the `f16` profile; `FIFI_EXL3_LAYER_MAJOR=off` is the rollback to the
//! token-major walk and `FIFI_EXL3_TALL_HILO=0` the rollback of the split alone.

use super::budget::{CpuMoeConfig, LayerMajor};
use super::settings::{FlintOptions, default_chunk_size};
use crate::EngError;

/// The facts the choice depends on.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OrderInputs<'a> {
    pub(crate) profile: &'a str,
    pub(crate) alloc_cache: bool,
    pub(crate) cpu_moe: &'a CpuMoeConfig,
    /// The configured prefill chunk is the default (eight rows).
    pub(crate) default_chunk: bool,
    /// The jobs pass's static reserve fits the working ceiling.
    pub(crate) jobs_admitted: bool,
}

impl<'a> OrderInputs<'a> {
    /// The facts of one open: the options' profile and chunk, and the three checks the caller
    /// makes of the host.
    pub(crate) fn for_options(
        options: &'a FlintOptions,
        cpu_moe: &'a CpuMoeConfig,
        alloc_cache: bool,
        jobs_admitted: bool,
    ) -> Self {
        Self {
            profile: &options.kv_profile,
            alloc_cache,
            cpu_moe,
            default_chunk: options.prefill_chunk_size == default_chunk_size(),
            jobs_admitted,
        }
    }
}

/// The order the engine opens with, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PrefillOrder {
    pub(crate) mode: LayerMajor,
    pub(crate) why: String,
    /// The switch was unset and the engine chose: only then may a jobs open that does not fit
    /// fall back to the token-major walk.
    pub(crate) defaulted: bool,
    /// The tall projections of the jobs pass split their activations (hi/lo). Only a jobs order
    /// can have it on.
    pub(crate) hilo: bool,
    /// The hi/lo state came from the default (unset, on with the jobs pass), not from the switch.
    pub(crate) hilo_defaulted: bool,
    /// The operator asked for the hi/lo split but the order is not the jobs pass, so it has no effect.
    pub(crate) hilo_inactive: bool,
    /// The open fell back from a defaulted jobs open to the token-major walk, so the load line must not read as an
    /// ordinary default token-major load.
    pub(crate) fell_back: bool,
}

impl PrefillOrder {
    /// The same order for the mode an open actually ran: the jobs retry fell back to the token-major walk, which has
    /// no split and whose source is the default fallback.
    pub(crate) fn with_mode(&self, mode: LayerMajor) -> PrefillOrder {
        if mode == self.mode {
            return self.clone();
        }
        PrefillOrder {
            mode,
            why: format!(
                "{} (fell back from {})",
                prefill_order_name(mode),
                prefill_order_name(self.mode)
            ),
            defaulted: true,
            hilo: false,
            hilo_defaulted: true,
            hilo_inactive: false,
            fell_back: true,
        }
    }

    /// The load-line tokens for the order: the source of the mode, and the hi/lo state with its source.
    pub(crate) fn load_suffix(&self) -> String {
        let source = |defaulted: bool| if defaulted { "default" } else { "explicit" };
        let mode_source = if self.fell_back {
            "default-fallback"
        } else {
            source(self.defaulted)
        };
        let mut line = format!(" prefill_order_source={mode_source}");
        if self.hilo {
            line.push_str(&format!(
                " tall_hilo=on tall_hilo_source={}",
                source(self.hilo_defaulted)
            ));
        } else if self.hilo_inactive {
            line.push_str(" tall_hilo=inactive");
        } else if self.mode == LayerMajor::Jobs {
            line.push_str(&format!(
                " tall_hilo=off tall_hilo_source={}",
                source(self.hilo_defaulted)
            ));
        }
        line
    }
}

/// How the start line names a prefill order.
pub(crate) fn prefill_order_name(mode: LayerMajor) -> &'static str {
    match mode {
        LayerMajor::Off => "token-major",
        LayerMajor::Tiles => "layer-major tiles",
        LayerMajor::Jobs => "layer-major jobs",
    }
}

/// What an engine whose switch is unset prefills with, and why: the job-table pass, unless one of
/// its conditions does not hold, in which case the token-major walk and the condition that failed.
/// The pass is measured and gated on the f16 profile only; it needs the allocation cache for its
/// tall frames; CPU experts keep the token-major walk; a configured chunk other than the default
/// keeps the walk the operator sized; and an engine whose reserve the pass would push past the
/// working ceiling opens as it always did.
pub(crate) fn default_layer_major(inputs: &OrderInputs<'_>) -> (LayerMajor, &'static str) {
    if inputs.profile != "f16" {
        (LayerMajor::Off, "the kv profile is not f16")
    } else if !inputs.alloc_cache {
        (LayerMajor::Off, "the allocation cache is off")
    } else if inputs.cpu_moe.enabled() {
        (LayerMajor::Off, "CPU experts are configured")
    } else if !inputs.default_chunk {
        (LayerMajor::Off, "prefill_chunk_size is not 8")
    } else if !inputs.jobs_admitted {
        (LayerMajor::Off, "its reserve does not fit this context")
    } else {
        (
            LayerMajor::Jobs,
            "the default: the job-table pass on the f16 profile",
        )
    }
}

/// CPU experts read the pinned host store token by token; neither layer-major pass stages their
/// layers, so the two are refused together before any device work starts. The text names the
/// switch and the options that conflict.
pub(crate) fn cpu_experts_allow(cpu_moe: &CpuMoeConfig, mode: LayerMajor) -> Result<(), EngError> {
    if cpu_moe.enabled() && mode != LayerMajor::Off {
        return Err(EngError::BadRequest(
            "FIFI_EXL3_LAYER_MAJOR (1 or jobs) cannot be combined with Flint CPU experts \
             (cpu_moe_offload_layers, cpu_moe_split_experts); set it to off or unset it, or turn \
             the CPU experts off"
                .into(),
        ));
    }
    Ok(())
}

/// The mixed q8k KV profile has no layer-major / jobs prefill arm: both passes
/// stage whole layers, but the q8k write path is only validated on the default
/// token-major launch. Refused at open, before any device work, naming the
/// switch and the profile.
pub(crate) fn q8k_layer_major_allow(profile: &str, mode: LayerMajor) -> Result<(), EngError> {
    if profile == "q8k" && mode != LayerMajor::Off {
        return Err(EngError::BadRequest(
            "FIFI_EXL3_LAYER_MAJOR (1 or jobs) has no q8k arm: use kv_profile q8k on the \
             token-major prefill (unset the switch or set it to off)"
                .into(),
        ));
    }
    Ok(())
}

/// The order for the parsed switches. An explicit `FIFI_EXL3_LAYER_MAJOR` is never softened: `1` or
/// `jobs` with CPU experts or the q8k profile is refused here. An unset switch takes the default of
/// [`default_layer_major`]. The hi/lo split is on with the jobs pass unless `FIFI_EXL3_TALL_HILO=0`;
/// `FIFI_EXL3_TALL_HILO=1` outside the jobs pass is reported inactive, not refused.
pub(crate) fn resolve_prefill_order(
    selected: Option<LayerMajor>,
    hilo: Option<bool>,
    inputs: &OrderInputs<'_>,
) -> Result<PrefillOrder, EngError> {
    let (mode, why, defaulted) = match selected {
        Some(mode) => {
            cpu_experts_allow(inputs.cpu_moe, mode)?;
            let name = match mode {
                LayerMajor::Off => "off",
                LayerMajor::Tiles => "1",
                LayerMajor::Jobs => "jobs",
            };
            (mode, format!("FIFI_EXL3_LAYER_MAJOR={name}"), false)
        }
        None => {
            let (mode, why) = default_layer_major(inputs);
            (mode, why.to_owned(), true)
        }
    };
    q8k_layer_major_allow(inputs.profile, mode)?;
    let jobs = mode == LayerMajor::Jobs;
    Ok(PrefillOrder {
        mode,
        why,
        defaulted,
        hilo: jobs && hilo.unwrap_or(true),
        hilo_defaulted: hilo.is_none(),
        hilo_inactive: !jobs && hilo == Some(true),
        fell_back: false,
    })
}

/// Whether a jobs open the engine chose by default failed for want of memory, so that the
/// token-major walk may be tried: a memory-plan refusal, or a device allocation that ran out.
/// Any other failure (a bad checkpoint, a bad switch) is not about the order and stays a failure.
///
/// The `Backend` arm only logs and fails fast: a native failure other than a clean allocation
/// refusal latches the process terminal, so the second open ends in the generic terminal
/// error. The `MemoryPlan` arm is the case a retry can serve.
pub(crate) fn retry_token_major(error: &EngError) -> bool {
    match error {
        EngError::MemoryPlan(_) => true,
        EngError::Backend { message, .. } => message.to_ascii_lowercase().contains("out of memory"),
        _ => false,
    }
}

/// Opens with the resolved order. When the switch was unset and the job-table pass was chosen, an open
/// that fails for want of memory is retried once on the token-major walk, with `log` told why; an explicit
/// value, and any other failure, is returned as it is. `open` is called with the order to open.
pub(crate) fn open_with_default_fallback<T>(
    order: &PrefillOrder,
    mut open: impl FnMut(LayerMajor) -> Result<T, EngError>,
    mut log: impl FnMut(&str),
) -> Result<T, EngError> {
    if !(order.defaulted && order.mode == LayerMajor::Jobs) {
        return open(order.mode);
    }
    match open(LayerMajor::Jobs) {
        Err(error) if retry_token_major(&error) => {
            log(&format!(
                "Flint prefill order: token-major (the job-table pass does not fit: {}); \
                 FIFI_EXL3_LAYER_MAJOR=jobs makes that a refusal and FIFI_EXL3_LAYER_MAJOR=off asks for the token-major walk",
                error.describe()
            ));
            open(LayerMajor::Off)
        }
        opened => opened,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs<'a>(profile: &'a str, cpu: &'a CpuMoeConfig) -> OrderInputs<'a> {
        OrderInputs {
            profile,
            alloc_cache: true,
            cpu_moe: cpu,
            default_chunk: true,
            jobs_admitted: true,
        }
    }

    fn none() -> CpuMoeConfig {
        CpuMoeConfig::new(0, 0, 0).unwrap()
    }

    fn cpu() -> CpuMoeConfig {
        CpuMoeConfig::new(2, 0, 0).unwrap()
    }

    #[test]
    fn an_unset_switch_defaults_to_jobs_only_where_every_condition_holds() {
        let (none, some) = (none(), cpu());
        assert_eq!(
            default_layer_major(&inputs("f16", &none)).0,
            LayerMajor::Jobs
        );
        // Each condition alone sends the engine back to the token-major walk and names itself.
        let cases: [(OrderInputs<'_>, &str); 6] = [
            (inputs("q8", &none), "kv profile"),
            (inputs("q8k", &none), "kv profile"),
            (
                OrderInputs {
                    alloc_cache: false,
                    ..inputs("f16", &none)
                },
                "allocation cache",
            ),
            (inputs("f16", &some), "CPU experts"),
            (
                OrderInputs {
                    default_chunk: false,
                    ..inputs("f16", &none)
                },
                "prefill_chunk_size",
            ),
            (
                OrderInputs {
                    jobs_admitted: false,
                    ..inputs("f16", &none)
                },
                "reserve",
            ),
        ];
        for (case, reason) in cases {
            let (mode, why) = default_layer_major(&case);
            assert_eq!(mode, LayerMajor::Off, "{case:?}");
            assert!(why.contains(reason), "{case:?}: {why}");
        }
    }

    /// The default of a Flint CUDA load is the job-table pass with the hi/lo split, both from the default.
    #[test]
    fn nothing_set_gives_jobs_with_the_hilo_split_on_both_from_the_default() {
        let none = none();
        let order = resolve_prefill_order(None, None, &inputs("f16", &none)).unwrap();
        assert_eq!((order.mode, order.hilo), (LayerMajor::Jobs, true));
        assert!(order.defaulted && order.hilo_defaulted && !order.hilo_inactive);
        assert_eq!(
            order.load_suffix(),
            " prefill_order_source=default tall_hilo=on tall_hilo_source=default"
        );
    }

    #[test]
    fn each_rollback_value_works_and_says_where_it_came_from() {
        let none = none();
        // The token-major rollback: no jobs pass, no split to speak of.
        let off =
            resolve_prefill_order(Some(LayerMajor::Off), None, &inputs("f16", &none)).unwrap();
        assert_eq!(
            (off.mode, off.hilo, off.defaulted),
            (LayerMajor::Off, false, false)
        );
        assert_eq!(off.load_suffix(), " prefill_order_source=explicit");
        // The split alone off: jobs with the F16 cast, the earlier jobs pass.
        let cast = resolve_prefill_order(None, Some(false), &inputs("f16", &none)).unwrap();
        assert_eq!(
            (cast.mode, cast.hilo, cast.hilo_defaulted),
            (LayerMajor::Jobs, false, false)
        );
        assert_eq!(
            cast.load_suffix(),
            " prefill_order_source=default tall_hilo=off tall_hilo_source=explicit"
        );
        // Explicit jobs keeps the split on by default; explicit 1 with the split asked for is inactive.
        let jobs =
            resolve_prefill_order(Some(LayerMajor::Jobs), None, &inputs("f16", &none)).unwrap();
        assert!(jobs.hilo && jobs.hilo_defaulted && !jobs.defaulted);
        let inactive =
            resolve_prefill_order(Some(LayerMajor::Off), Some(true), &inputs("f16", &none))
                .unwrap();
        assert!(!inactive.hilo && inactive.hilo_inactive);
        assert_eq!(
            inactive.load_suffix(),
            " prefill_order_source=explicit tall_hilo=inactive"
        );
        // The tile pass has no split either.
        let tiles =
            resolve_prefill_order(Some(LayerMajor::Tiles), None, &inputs("f16", &none)).unwrap();
        assert!(!tiles.hilo);
    }

    #[test]
    fn the_defaults_fall_back_and_name_the_reason_off_the_f16_profile() {
        let (none, some) = (none(), cpu());
        for (profile, cpu, cache) in [
            ("q8", &none, true),
            ("q8k", &none, true),
            ("f16", &some, true),
            ("f16", &none, false),
        ] {
            let order = resolve_prefill_order(
                None,
                None,
                &OrderInputs {
                    alloc_cache: cache,
                    ..inputs(profile, cpu)
                },
            )
            .unwrap();
            assert_eq!(
                (order.mode, order.hilo),
                (LayerMajor::Off, false),
                "{profile} {cache}"
            );
            assert!(order.defaulted && !order.why.is_empty(), "{order:?}");
        }
    }

    #[test]
    fn an_explicit_layer_major_is_never_softened_and_the_refusals_hold() {
        let (none, some) = (none(), cpu());
        for mode in [LayerMajor::Tiles, LayerMajor::Jobs] {
            let refused = resolve_prefill_order(Some(mode), None, &inputs("f16", &some));
            assert!(
                matches!(&refused, Err(EngError::BadRequest(text))
                    if text.contains("FIFI_EXL3_LAYER_MAJOR") && text.contains("cpu_moe_offload_layers")),
                "{mode:?}: {refused:?}"
            );
            let refused = resolve_prefill_order(Some(mode), None, &inputs("q8k", &none));
            assert!(
                matches!(&refused, Err(EngError::BadRequest(text)) if text.contains("q8k")),
                "{mode:?}: {refused:?}"
            );
            // The profile, the cache, the chunk and the reserve do not soften an explicit value.
            let order = resolve_prefill_order(
                Some(mode),
                None,
                &OrderInputs {
                    alloc_cache: false,
                    default_chunk: false,
                    jobs_admitted: false,
                    ..inputs("q8", &none)
                },
            )
            .unwrap();
            assert_eq!(order.mode, mode);
            assert!(!order.defaulted);
        }
        // `off` is the rollback on every profile, with CPU experts too.
        for (profile, cpu) in [
            ("f16", &none),
            ("q8", &none),
            ("q8k", &none),
            ("f16", &some),
        ] {
            let order =
                resolve_prefill_order(Some(LayerMajor::Off), None, &inputs(profile, cpu)).unwrap();
            assert_eq!(order.mode, LayerMajor::Off);
            assert!(order.why.contains("=off") && !order.defaulted, "{order:?}");
        }
    }

    #[test]
    fn q8k_refuses_both_layer_major_arms_at_open() {
        for mode in [LayerMajor::Tiles, LayerMajor::Jobs] {
            let error = q8k_layer_major_allow("q8k", mode).expect_err("q8k has no layer-major arm");
            let text = format!("{error:?}");
            assert!(
                text.contains("q8k")
                    && text.contains("FIFI_EXL3_LAYER_MAJOR")
                    && text.contains("token-major"),
                "{text}"
            );
        }
        assert!(q8k_layer_major_allow("q8k", LayerMajor::Off).is_ok());
        assert!(q8k_layer_major_allow("f16", LayerMajor::Jobs).is_ok());
        assert!(q8k_layer_major_allow("q8", LayerMajor::Tiles).is_ok());
    }

    #[test]
    fn a_defaulted_jobs_open_retries_on_a_memory_shortage_only() {
        assert!(retry_token_major(&EngError::MemoryPlan(
            "need 5 GiB".into()
        )));
        assert!(retry_token_major(&EngError::Backend {
            operation: "alloc",
            message: "CUDA error: Out Of Memory".into()
        }));
        assert!(!retry_token_major(&EngError::Backend {
            operation: "launch",
            message: "illegal address".into()
        }));
        assert!(!retry_token_major(&EngError::Checkpoint("bad pack".into())));
        assert!(!retry_token_major(&EngError::BadRequest(
            "bad switch".into()
        )));
    }

    #[test]
    fn the_inputs_come_from_the_options_and_the_host_checks() {
        let none = none();
        let mut options = FlintOptions::cuda_default();
        let base = OrderInputs::for_options(&options, &none, true, true);
        assert_eq!(base.profile, "f16");
        assert!(
            base.alloc_cache && base.default_chunk && base.jobs_admitted && !base.cpu_moe.enabled()
        );
        assert_eq!(
            resolve_prefill_order(None, None, &base).unwrap().mode,
            LayerMajor::Jobs
        );
        options.prefill_chunk_size = 4;
        let chunk = OrderInputs::for_options(&options, &none, true, true);
        assert!(!chunk.default_chunk);
        assert_eq!(
            resolve_prefill_order(None, None, &chunk).unwrap().mode,
            LayerMajor::Off
        );
        options.prefill_chunk_size = 8;
        options.kv_profile = "q8k".into();
        let q8k = OrderInputs::for_options(&options, &none, true, true);
        assert_eq!(
            resolve_prefill_order(None, None, &q8k).unwrap().mode,
            LayerMajor::Off
        );
        assert!(resolve_prefill_order(Some(LayerMajor::Jobs), None, &q8k).is_err());
    }

    #[test]
    fn the_order_is_named_for_the_start_line() {
        assert_eq!(prefill_order_name(LayerMajor::Off), "token-major");
        assert_eq!(prefill_order_name(LayerMajor::Jobs), "layer-major jobs");
    }

    /// The fallback an open actually ran reads as the default token-major walk: no split, source
    /// default, and the reason names both orders; an unchanged mode is the same order.
    #[test]
    fn a_fallback_order_is_the_defaulted_token_major_walk_without_the_split() {
        let jobs = resolve_prefill_order(None, None, &inputs("f16", &none())).unwrap();
        assert!(jobs.hilo && jobs.defaulted);
        assert_eq!(jobs.with_mode(LayerMajor::Jobs), jobs);
        let fell = jobs.with_mode(LayerMajor::Off);
        assert_eq!(fell.mode, LayerMajor::Off);
        assert!(fell.defaulted && !fell.hilo && !fell.hilo_inactive);
        assert!(
            fell.why.contains("fell back from layer-major jobs"),
            "{}",
            fell.why
        );
        assert!(fell.fell_back && !jobs.fell_back);
        assert_eq!(fell.load_suffix(), " prefill_order_source=default-fallback");
    }

    /// `open_resolved` hands the successful attempt's hi/lo state to the kernels before any device
    /// work and retains that same attempt's order on the executor for runtime settings.
    #[test]
    fn the_open_hands_the_resolved_hilo_state_to_the_kernels() {
        let source = include_str!("cuda_executor.rs");
        let start = source
            .find("    fn open_resolved(")
            .expect("open_resolved exists");
        let body = &source[start..start + 1600];
        let set = body
            .find("super::settings::set_tall_hilo(hilo);")
            .expect("open_resolved calls set_tall_hilo(hilo)");
        let device = body
            .find("CudaContext::")
            .expect("open_resolved opens the context");
        assert!(set < device, "the state is set before any device work");
        assert!(
            body.contains("let PrefillOrder { mode, hilo, .. } = order;"),
            "hilo comes from the resolved order"
        );
        for wrong in [
            "set_tall_hilo(!hilo)",
            "set_tall_hilo(true)",
            "set_tall_hilo(false)",
        ] {
            assert!(!source.contains(wrong), "{wrong}");
        }
        let open = source.find("    fn open_with(").expect("open_with exists");
        assert!(
            source[open..start].contains("order.with_mode(mode),"),
            "each open attempt receives its own order"
        );
        assert!(
            source[start..].contains("served_order: order,"),
            "the successful executor retains its own order"
        );
    }

    fn order(mode: LayerMajor, defaulted: bool) -> PrefillOrder {
        PrefillOrder {
            mode,
            why: String::new(),
            defaulted,
            hilo: false,
            hilo_defaulted: true,
            hilo_inactive: false,
            fell_back: false,
        }
    }

    /// Runs the fallback with a scripted open: `script` answers the calls in order.
    fn drive(
        order: &PrefillOrder,
        script: Vec<Result<u32, EngError>>,
    ) -> (Result<u32, EngError>, Vec<LayerMajor>, Vec<String>) {
        let mut answers = script.into_iter();
        let mut calls = Vec::new();
        let mut lines = Vec::new();
        let result = open_with_default_fallback(
            order,
            |mode| {
                calls.push(mode);
                answers
                    .next()
                    .expect("the open was called more often than scripted")
            },
            |line| lines.push(line.to_owned()),
        );
        (result, calls, lines)
    }

    #[test]
    fn a_defaulted_jobs_open_that_does_not_fit_is_retried_on_the_token_major_walk() {
        let jobs = order(LayerMajor::Jobs, true);
        let (result, calls, lines) = drive(
            &jobs,
            vec![Err(EngError::MemoryPlan("need 5 GiB".into())), Ok(7)],
        );
        assert_eq!(result.unwrap(), 7);
        assert_eq!(calls, [LayerMajor::Jobs, LayerMajor::Off]);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(
            lines[0].contains("the job-table pass does not fit: need 5 GiB"),
            "{lines:?}"
        );
        assert!(lines[0].contains("FIFI_EXL3_LAYER_MAJOR=jobs"), "{lines:?}");
        // The retry's own failure is the failure returned, after exactly two calls.
        let (result, calls, _) = drive(
            &jobs,
            vec![
                Err(EngError::MemoryPlan("first".into())),
                Err(EngError::MemoryPlan("second".into())),
            ],
        );
        assert!(matches!(result, Err(EngError::MemoryPlan(m)) if m == "second"));
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn a_defaulted_jobs_open_that_succeeds_or_fails_otherwise_is_not_retried() {
        let jobs = order(LayerMajor::Jobs, true);
        let (result, calls, lines) = drive(&jobs, vec![Ok(1)]);
        assert_eq!(result.unwrap(), 1);
        assert_eq!(calls, [LayerMajor::Jobs]);
        assert!(lines.is_empty());
        for error in [
            EngError::Checkpoint("bad pack".into()),
            EngError::BadRequest("bad switch".into()),
            EngError::Backend {
                operation: "launch",
                message: "illegal address".into(),
            },
        ] {
            let (result, calls, lines) = drive(&jobs, vec![Err(error)]);
            assert!(result.is_err());
            assert_eq!(calls, [LayerMajor::Jobs]);
            assert!(lines.is_empty());
        }
    }

    #[test]
    fn an_explicit_order_and_a_defaulted_fallback_open_once_as_resolved() {
        for mode in [LayerMajor::Off, LayerMajor::Tiles, LayerMajor::Jobs] {
            let explicit = order(mode, false);
            let (result, calls, lines) =
                drive(&explicit, vec![Err(EngError::MemoryPlan("need".into()))]);
            assert!(matches!(result, Err(EngError::MemoryPlan(_))), "{mode:?}");
            assert_eq!(calls, [mode], "{mode:?}");
            assert!(lines.is_empty(), "{mode:?}");
        }
        // A default that already resolved to the token-major walk has nothing to fall back to.
        let off = order(LayerMajor::Off, true);
        let (result, calls, lines) = drive(&off, vec![Err(EngError::MemoryPlan("need".into()))]);
        assert!(result.is_err());
        assert_eq!(calls, [LayerMajor::Off]);
        assert!(lines.is_empty());
    }
}
