//! The effective configuration: every resolved value with where it came
//! from.
//!
//! Resolution itself stays exactly where it always was —
//! `Q36Settings::new` and its per-setting rules, with their current
//! per-key precedence (a config member, else one named environment
//! fallback where a setting has one, else a default). This module adds
//! no rule of its own: given the options, the environment fallback, and
//! the resolved settings, it names each principal value's origin. That
//! keeps preflight, startup, and this view in agreement by construction
//! — there is one resolver, and three renderers of its result.
//!
//! Origins are deliberately coarse and honest. `Derived` is only for
//! values computed from other resolved settings by the resolver itself
//! (today: the MLX prefill reserve and the dense session cap).
//! `Ignored` is for members the active backend reads past, reported
//! from the resolver's own `ignored_mlx_keys` list. One exception is
//! computed here rather than disclosed by the resolver: `compile_program_limit`
//! refuses an explicit CUDA value instead of ignoring it, so it can never
//! appear in `ignored_mlx_keys`; its row reports `IgnoredOnThisBackend` on
//! CUDA by construction (only the default is CUDA-reachable).

use serde::Serialize;

use crate::family::qwen36::settings::{Backend, Q36Options, Q36Settings};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// Set in the configuration file (or the equivalent CLI candidate).
    ConfigFile,
    /// Read from the setting's named environment variable.
    Environment,
    /// Computed by the resolver from other resolved settings.
    Derived,
    /// The resolver's built-in default for the active backend.
    Default,
    /// Present in the file but not read by the active backend.
    IgnoredOnThisBackend,
}

impl Origin {
    pub fn name(self) -> &'static str {
        match self {
            Origin::ConfigFile => "config file",
            Origin::Environment => "environment",
            Origin::Derived => "derived",
            Origin::Default => "default",
            Origin::IgnoredOnThisBackend => "ignored on this backend",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EffectiveSetting {
    pub key: &'static str,
    /// The resolved value, rendered exactly (bytes stay bytes).
    pub value: String,
    pub origin: Origin,
}

fn row(key: &'static str, value: String, origin: Origin) -> EffectiveSetting {
    EffectiveSetting { key, value, origin }
}

fn present<T>(value: &Option<T>) -> Origin {
    if value.is_some() {
        Origin::ConfigFile
    } else {
        Origin::Default
    }
}

/// The principal Qwen3.6 settings with origins. `env_cache_limit` is the
/// `Q36_CACHE_LIMIT` fallback exactly as the factory reads it; callers
/// in a diagnostic process read the real environment, tests inject.
pub(crate) fn q36_effective(
    options: &Q36Options,
    env_cache_limit: Option<&str>,
    backend: Backend,
    settings: &Q36Settings,
) -> Vec<EffectiveSetting> {
    use crate::family::qwen36::settings::{activation_name, kv_dtype_name, weight_format_name};
    let ignored = |key: &str| settings.ignored_mlx_keys.contains(&key);
    let on_arm = |key: &str, origin: Origin| {
        if ignored(key) {
            Origin::IgnoredOnThisBackend
        } else {
            origin
        }
    };
    let dense = settings.spec.feed_forward.is_some();

    let cache_origin = if options.cache_limit_bytes.is_some() {
        Origin::ConfigFile
    } else if backend == Backend::Mlx && env_cache_limit.is_some() {
        Origin::Environment
    } else {
        Origin::Default
    };
    let reserve_origin = if options.prefill_session_reserve_bytes.is_some() {
        Origin::ConfigFile
    } else if backend == Backend::Mlx && options.prefill_budget_bytes.is_some() {
        Origin::Derived
    } else {
        Origin::Default
    };
    let cap_origin = if options.session_cap_bytes.is_some() {
        Origin::ConfigFile
    } else if dense && backend == Backend::Mlx {
        Origin::Derived
    } else {
        Origin::Default
    };

    vec![
        row(
            "weight_format",
            weight_format_name(settings.weights).to_string(),
            present(&options.weight_format),
        ),
        row(
            "gguf_sha256",
            settings
                .gguf_sha256
                .clone()
                .unwrap_or_else(|| "off".to_string()),
            present(&options.gguf_sha256),
        ),
        row(
            "context_limit",
            settings.context_limit.to_string(),
            present(&options.context_limit),
        ),
        row(
            "kv_dtype",
            kv_dtype_name(settings.kv).to_string(),
            present(&options.kv_dtype),
        ),
        row(
            "activation_dtype",
            activation_name(settings.activation).to_string(),
            present(&options.activation_dtype),
        ),
        row(
            "cache_limit_bytes",
            settings.cache_limit.to_string(),
            on_arm("cache_limit_bytes", cache_origin),
        ),
        row(
            "session_cap_bytes",
            settings.session_cap.to_string(),
            on_arm("session_cap_bytes", cap_origin),
        ),
        row(
            "prefill_budget_bytes",
            settings
                .prefill_budget
                .map_or_else(|| "off".to_string(), |b| b.to_string()),
            present(&options.prefill_budget_bytes),
        ),
        row(
            "prefill_session_reserve_bytes",
            settings.prefill_reserve.to_string(),
            reserve_origin,
        ),
        row(
            "prefill_chunk",
            settings.prefill_chunk.to_string(),
            on_arm("prefill_chunk", present(&options.prefill_chunk)),
        ),
        row(
            "max_seqs",
            settings.max_seqs.to_string(),
            present(&options.max_seqs),
        ),
        row(
            "prefix_store_entries",
            settings
                .prefix_store_entries
                .map_or_else(|| "default (6)".to_string(), |n| n.to_string()),
            on_arm(
                "prefix_store_entries",
                present(&options.prefix_store_entries),
            ),
        ),
        row(
            "compile",
            settings.compile.to_string(),
            on_arm("compile", present(&options.compile)),
        ),
        row(
            "compile_program_limit",
            settings.compile_program_limit.get().to_string(),
            if backend == Backend::Cuda {
                Origin::IgnoredOnThisBackend
            } else {
                present(&options.compile_program_limit)
            },
        ),
        row(
            "capture",
            settings.capture.to_string(),
            present(&options.capture),
        ),
    ]
}

/// The principal Flint settings with origins, in the same three-part
/// shape as the Qwen3.6 rows: the `flint` config object (a field set
/// there versus the CUDA default), the family's environment switches
/// (`FlintEnv`, read by the caller so diagnostic processes and tests
/// inject rather than capture), and the resolved values the switches
/// produce. Flint's MLX path ignores the `flint` object; these rows are
/// the CUDA view.
pub(crate) fn flint_effective(
    options: &crate::family::flint::settings::FlintOptions,
    env: &crate::family::flint::budget::FlintEnv,
    served: Option<&crate::family::flint::prefill_order::PrefillOrder>,
) -> Vec<EffectiveSetting> {
    use crate::family::flint::{
        budget::{BankRequest, ExpertCacheMode, LayerMajor},
        settings::FlintOptions,
    };

    // A config field counts as set when it differs from the no-`flint`
    // default; FlintOptions has no Option fields because a present
    // `flint` object must stay explicit.
    let defaults = FlintOptions::cuda_default();
    let from_config = |value: &str, default: &str| {
        if value == default {
            Origin::Default
        } else {
            Origin::ConfigFile
        }
    };

    let mut rows = vec![
        row(
            "context_limit",
            options.context_limit.to_string(),
            if options.context_limit == defaults.context_limit {
                Origin::Default
            } else {
                Origin::ConfigFile
            },
        ),
        row(
            "kv_profile",
            options.kv_profile.clone(),
            from_config(&options.kv_profile, &defaults.kv_profile),
        ),
        row(
            "prefill_chunk_size",
            options.prefill_chunk_size.to_string(),
            if options.prefill_chunk_size == defaults.prefill_chunk_size {
                Origin::Default
            } else {
                Origin::ConfigFile
            },
        ),
        row(
            "cpu_moe_offload_layers",
            options.cpu_moe_offload_layers.to_string(),
            if options.cpu_moe_offload_layers == 0 {
                Origin::Default
            } else {
                Origin::ConfigFile
            },
        ),
        row(
            "cpu_moe_split_experts",
            options.cpu_moe_split_experts.to_string(),
            if options.cpu_moe_split_experts == 0 {
                Origin::Default
            } else {
                Origin::ConfigFile
            },
        ),
        row(
            "cpu_moe_threads",
            if options.cpu_moe_threads == 0 {
                "default (6)".to_owned()
            } else {
                options.cpu_moe_threads.to_string()
            },
            if options.cpu_moe_threads == 0 {
                Origin::Default
            } else {
                Origin::ConfigFile
            },
        ),
    ];
    // The environment switches resolve or refuse; an invalid value shows
    // as the refusal text rather than a made-up setting.
    let switch = |set: bool| {
        if set {
            Origin::Environment
        } else {
            Origin::Default
        }
    };
    let mode = env
        .expert_cache_mode()
        .map(ExpertCacheMode::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|error| format!("refused: {error:?}"));
    rows.push(row(
        "expert_cache",
        mode,
        switch(env.expert_cache.is_some()),
    ));
    let request = env.bank_request().map_or_else(
        |error| format!("refused: {error:?}"),
        |request| match request {
            BankRequest::Fit => "fit".to_owned(),
            BankRequest::Exactly(count) => format!("exactly {count}"),
        },
    );
    rows.push(row(
        "expert_slots",
        request,
        switch(env.expert_slots.is_some()),
    ));
    // The prefill order resolves the way the executor open resolves it: the job-table pass with the
    // hi/lo split by default on the f16 profile, the operator's words over that.
    let cpu_moe = crate::family::flint::budget::CpuMoeConfig::new(
        options.cpu_moe_offload_layers,
        options.cpu_moe_split_experts,
        options.cpu_moe_threads,
    )
    .unwrap_or_default();
    // A running engine reports the order its open actually served (a retry reads as the token-major walk);
    // a diagnostic with no engine resolves it from the environment, assuming the jobs reserve is admitted.
    let resolved = match served {
        Some(order) => Ok(order.clone()),
        None => env.prefill_order(options, &cpu_moe),
    };
    let (layer_major, hilo) = match resolved {
        Ok(order) => {
            let name = match order.mode {
                LayerMajor::Off => "off (token-major)",
                LayerMajor::Tiles => "tiles (1)",
                LayerMajor::Jobs => "jobs",
            };
            let source = if order.fell_back {
                "default-fallback"
            } else if order.defaulted {
                "default"
            } else {
                "explicit"
            };
            let hilo = if order.hilo {
                "on".to_owned()
            } else if order.hilo_inactive {
                "inactive (needs the jobs pass)".to_owned()
            } else {
                "off".to_owned()
            };
            (
                format!("{name} ({source}: {})", order.why),
                (hilo, order.hilo_defaulted),
            )
        }
        Err(error) => (format!("refused: {error:?}"), ("refused".to_owned(), false)),
    };
    rows.push(row(
        "layer_major",
        layer_major,
        switch(env.layer_major.is_some()),
    ));
    rows.push(row("tall_hilo", hilo.0, switch(!hilo.1)));
    rows.push(row(
        "alloc_cache",
        if env.alloc_cache_enabled() {
            "on"
        } else {
            "off"
        }
        .to_owned(),
        switch(env.alloc_cache.is_some()),
    ));
    rows
}

#[cfg(test)]
mod tests {
    use super::{Origin, q36_effective};
    use crate::family::qwen36::settings::{Backend, Q36Env, Q36Options, Q36Settings};
    use std::path::Path;

    fn resolve(options: &Q36Options, backend: Backend) -> Q36Settings {
        Q36Settings::new(
            Path::new("m"),
            Path::new("t"),
            Some(options),
            &Q36Env::default(),
            backend,
        )
        .expect("resolves")
    }

    fn origin(rows: &[super::EffectiveSetting], key: &str) -> Origin {
        rows.iter()
            .find(|r| r.key == key)
            .unwrap_or_else(|| panic!("no row {key}"))
            .origin
    }

    #[test]
    fn q36_program_budget_reports_the_resolved_limit_and_origin() {
        for (limit, expected, source) in [
            (None, "8", Origin::Default),
            (Some(12), "12", Origin::ConfigFile),
        ] {
            let options = Q36Options {
                cache_limit_bytes: Some(1024),
                compile_program_limit: limit,
                ..Q36Options::default()
            };
            let settings = resolve(&options, Backend::Mlx);
            let rows = q36_effective(&options, None, Backend::Mlx, &settings);
            let row = rows
                .iter()
                .find(|row| row.key == "compile_program_limit")
                .unwrap();
            assert_eq!(row.value, expected);
            assert_eq!(row.origin, source);
        }
        let options = Q36Options::default();
        let settings = resolve(&options, Backend::Cuda);
        let rows = q36_effective(&options, None, Backend::Cuda, &settings);
        assert_eq!(
            origin(&rows, "compile_program_limit"),
            Origin::IgnoredOnThisBackend
        );
    }

    #[test]
    fn q36_content_pin_reports_the_normalized_value_and_its_origin() {
        for pin in [None, Some("A".repeat(64))] {
            let options = Q36Options {
                gguf_sha256: pin.clone(),
                ..Q36Options::default()
            };
            let settings = resolve(&options, Backend::Cuda);
            let rows = q36_effective(&options, None, Backend::Cuda, &settings);
            let row = rows.iter().find(|row| row.key == "gguf_sha256").unwrap();
            assert_eq!(
                row.value,
                if pin.is_some() {
                    "a".repeat(64)
                } else {
                    "off".to_string()
                }
            );
            assert_eq!(
                row.origin,
                if pin.is_some() {
                    Origin::ConfigFile
                } else {
                    Origin::Default
                }
            );
        }
    }

    #[test]
    fn flint_rows_name_config_env_and_default_origins() {
        use crate::family::flint::budget::FlintEnv;
        use crate::family::flint::settings::FlintOptions;

        let rows =
            super::flint_effective(&FlintOptions::cuda_default(), &FlintEnv::default(), None);
        assert_eq!(origin(&rows, "context_limit"), Origin::Default);
        let context = rows.iter().find(|r| r.key == "context_limit").unwrap();
        assert_eq!(context.value, "131072");
        assert_eq!(origin(&rows, "kv_profile"), Origin::Default);
        // The default prefill order of a Flint load: the job-table pass with the hi/lo split, from the defaults.
        assert_eq!(origin(&rows, "layer_major"), Origin::Default);
        let major = rows.iter().find(|r| r.key == "layer_major").unwrap();
        assert!(major.value.starts_with("jobs (default"), "{}", major.value);
        assert_eq!(origin(&rows, "tall_hilo"), Origin::Default);
        assert!(rows.iter().any(|r| r.key == "tall_hilo" && r.value == "on"));
        // The alloc cache is default-on; the row says so without a var set.
        assert_eq!(origin(&rows, "alloc_cache"), Origin::Default);
        assert!(
            rows.iter()
                .any(|r| r.key == "alloc_cache" && r.value == "on")
        );

        let mut options = FlintOptions::cuda_default();
        options.kv_profile = "q8".to_owned();
        options.cpu_moe_threads = 4;
        let env = FlintEnv {
            expert_slots: Some("226".to_owned()),
            layer_major: Some(std::ffi::OsString::from("jobs")),
            ..FlintEnv::default()
        };
        let rows = super::flint_effective(&options, &env, None);
        assert_eq!(origin(&rows, "kv_profile"), Origin::ConfigFile);
        assert_eq!(origin(&rows, "cpu_moe_threads"), Origin::ConfigFile);
        assert_eq!(origin(&rows, "expert_slots"), Origin::Environment);
        let slots = rows.iter().find(|r| r.key == "expert_slots").unwrap();
        assert_eq!(slots.value, "exactly 226");
        assert_eq!(origin(&rows, "layer_major"), Origin::Environment);
        let major = rows.iter().find(|r| r.key == "layer_major").unwrap();
        assert!(major.value.starts_with("jobs (explicit"), "{}", major.value);

        // A refused switch value renders as the refusal, not a setting.
        let env = FlintEnv {
            layer_major: Some(std::ffi::OsString::from("yes")),
            ..FlintEnv::default()
        };
        let rows = super::flint_effective(&FlintOptions::cuda_default(), &env, None);
        let major = rows.iter().find(|r| r.key == "layer_major").unwrap();
        assert!(major.value.starts_with("refused:"), "{}", major.value);
    }

    /// A running engine reports the order its open served: a defaulted jobs open that fell back to the token-major
    /// walk reads as that walk, with no split, not as the environment's default.
    #[test]
    fn a_served_fallback_is_reported_as_the_token_major_walk() {
        use crate::family::flint::budget::{FlintEnv, LayerMajor};
        use crate::family::flint::prefill_order::{OrderInputs, resolve_prefill_order};
        use crate::family::flint::settings::FlintOptions;
        let options = FlintOptions::cuda_default();
        let none = crate::family::flint::budget::CpuMoeConfig::default();
        let jobs = resolve_prefill_order(
            None,
            None,
            &OrderInputs::for_options(&options, &none, true, true),
        )
        .unwrap();
        let fell = jobs.with_mode(LayerMajor::Off);
        let rows = super::flint_effective(&options, &FlintEnv::default(), Some(&fell));
        let major = rows.iter().find(|r| r.key == "layer_major").unwrap();
        assert!(
            major
                .value
                .starts_with("off (token-major) (default-fallback"),
            "{}",
            major.value
        );
        assert!(
            rows.iter()
                .any(|r| r.key == "tall_hilo" && r.value == "off")
        );
        let rows = super::flint_effective(&options, &FlintEnv::default(), Some(&jobs));
        assert!(rows.iter().any(|r| r.key == "tall_hilo" && r.value == "on"));
    }

    /// The Flint prefill order and its hi/lo split are Flint settings: a Qwen3.6 load reports neither,
    /// and no Qwen3.6 source reads the two switches.
    #[test]
    fn a_qwen36_load_has_no_flint_prefill_order() {
        let options = Q36Options {
            cache_limit_bytes: Some(4_294_967_296),
            context_limit: Some(262_144),
            prefill_budget_bytes: Some(32_212_254_720),
            ..Q36Options::default()
        };
        let settings = resolve(&options, Backend::Mlx);
        let rows = q36_effective(&options, None, Backend::Mlx, &settings);
        for key in ["layer_major", "tall_hilo"] {
            assert!(rows.iter().all(|r| r.key != key), "{key} is a Flint row");
        }
        let qwen = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/family/qwen36");
        let mut stack = vec![qwen];
        let mut seen = 0;
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    seen += 1;
                    let source = std::fs::read_to_string(&path).unwrap();
                    for needle in [
                        "FIFI_EXL3_TALL_HILO",
                        "FIFI_EXL3_LAYER_MAJOR",
                        "TALL_HILO_VAR",
                    ] {
                        assert!(
                            !source.contains(needle),
                            "{} reads {needle}",
                            path.display()
                        );
                    }
                }
            }
        }
        assert!(seen > 10, "the Qwen3.6 sources were not found");
    }

    #[test]
    fn origins_distinguish_file_derived_and_default() {
        let options = Q36Options {
            cache_limit_bytes: Some(4_294_967_296),
            context_limit: Some(262_144),
            prefill_budget_bytes: Some(32_212_254_720),
            kv_dtype: Some("q8".into()),
            ..Q36Options::default()
        };
        let settings = resolve(&options, Backend::Mlx);
        let rows = q36_effective(&options, None, Backend::Mlx, &settings);
        assert_eq!(origin(&rows, "context_limit"), Origin::ConfigFile);
        assert_eq!(origin(&rows, "cache_limit_bytes"), Origin::ConfigFile);
        assert_eq!(
            origin(&rows, "prefill_session_reserve_bytes"),
            Origin::Derived
        );
        assert_eq!(origin(&rows, "activation_dtype"), Origin::Default);
        assert_eq!(origin(&rows, "session_cap_bytes"), Origin::Default);
        let reserve = rows
            .iter()
            .find(|r| r.key == "prefill_session_reserve_bytes")
            .expect("row");
        assert_eq!(reserve.value, settings.prefill_reserve.to_string());
    }

    #[test]
    fn the_environment_fallback_and_ignored_keys_are_named() {
        let options = Q36Options {
            session_cap_bytes: Some(1000),
            compile: Some(true),
            ..Q36Options::default()
        };
        let env = Q36Env {
            cache_limit: Some("2048".to_string()),
        };
        let settings = Q36Settings::new(
            Path::new("m"),
            Path::new("t"),
            Some(&options),
            &env,
            Backend::Mlx,
        )
        .expect("resolves via env");
        let rows = q36_effective(&options, Some("2048"), Backend::Mlx, &settings);
        assert_eq!(origin(&rows, "cache_limit_bytes"), Origin::Environment);

        let cuda = resolve(&options, Backend::Cuda);
        let rows = q36_effective(&options, None, Backend::Cuda, &cuda);
        assert_eq!(
            origin(&rows, "session_cap_bytes"),
            Origin::IgnoredOnThisBackend
        );
        assert_eq!(origin(&rows, "compile"), Origin::IgnoredOnThisBackend);
    }
}
