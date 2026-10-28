use super::check_config;
use super::prefix_reuse_env;

const BASE: &[u8] = include_bytes!("../../../testdata/flint/config.json");
const MLX: &[u8] = include_bytes!("../../../testdata/flint/mlx-config.json");

#[test]
fn prefix_reuse_opt_in_polarity_is_fail_safe_absent() {
    // Opt-in: ONLY the enable words turn it on; absence and every other
    // shape stay off, so a mis-set or missing variable can never arm the
    // unqualified route.
    assert!(!prefix_reuse_env(Err(std::env::VarError::NotPresent)));
    assert!(!prefix_reuse_env(Err(std::env::VarError::NotUnicode(
        std::ffi::OsString::new()
    ))));
    assert!(!prefix_reuse_env(Ok(String::new())));
    for word in ["0", "no", "off", "Yes", "true=1"] {
        assert!(!prefix_reuse_env(Ok(word.to_string())), "{word:?}");
    }
    // " ON\n" enables: enable words are trimmed and case-folded (the
    // convention every family switch documents).
    for word in ["1", "true", "on", " ON ", " ON\n"] {
        assert!(prefix_reuse_env(Ok(word.to_string())), "{word:?}");
    }
}

#[test]
fn accepts_both_pinned_configurations() {
    for bytes in [BASE, MLX] {
        check_config(bytes).expect("pinned Flint metadata");
    }
}

fn assert_field(config: &serde_json::Value, field: &str) {
    match check_config(&serde_json::to_vec(config).unwrap()) {
        Err(crate::EngError::Checkpoint(message)) => assert!(
            message.contains(field),
            "expected field {field}, got {message}"
        ),
        Err(other) => panic!("expected checkpoint error, got {other:?}"),
        Ok(()) => panic!("altered Flint field {field} was accepted"),
    }
}

#[test]
fn rejects_altered_inference_fields() {
    let changes = serde_json::json!({
        "model_type": "other",
        "architectures": ["OtherForCausalLM"],
        "attention_bias": true,
        "attention_dropout": 0.1,
        "hidden_act": "relu",
        "hidden_size": 4096,
        "intermediate_size": 6145,
        "head_dim": 129,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 513,
        "shared_expert_intermediate_size": 1025,
        "decoder_sparse_step": 2,
        "sliding_window": 512,
        "max_position_embeddings": 131072,
        "vocab_size": 150000,
        "bos_token_id": 0,
        "pad_token_id": 0,
        "eos_token_id": [150001],
        "rms_norm_eps": 0.01,
        "rope_theta": 20000,
        "router_score_function": "softmax",
        "norm_topk_prob": false,
        "softmax_type": "softmax",
        "tie_word_embeddings": true,
        "use_cache": false,
        "use_sliding_window": false,
        "output_router_logits": true
    });
    for bytes in [BASE, MLX] {
        for (field, changed) in changes.as_object().unwrap() {
            let mut config: serde_json::Value = serde_json::from_slice(bytes).unwrap();
            config[field] = changed.clone();
            assert_field(&config, field);
        }
    }
}

#[test]
fn rejects_missing_and_wrongly_typed_dimensions() {
    for changed in [
        serde_json::json!(2048.0),
        serde_json::json!("2048"),
        serde_json::Value::Null,
    ] {
        let mut config: serde_json::Value = serde_json::from_slice(BASE).unwrap();
        config["hidden_size"] = changed;
        assert_field(&config, "hidden_size");
    }
    let mut config: serde_json::Value = serde_json::from_slice(BASE).unwrap();
    config.as_object_mut().unwrap().remove("hidden_size");
    assert_field(&config, "hidden_size");
}

#[test]
fn rejects_duplicate_keys_and_non_object_documents() {
    let duplicate = br#"{"model_type":"flint","model_type":"flint"}"#;
    match check_config(duplicate) {
        Err(crate::EngError::Checkpoint(message)) => {
            assert!(message.contains("duplicate"), "{message}")
        }
        other => panic!("expected duplicate-key refusal, got {other:?}"),
    }
    for json in [b"null".as_slice(), b"[]", b"false", b"{"] {
        assert!(matches!(
            check_config(json),
            Err(crate::EngError::Checkpoint(_))
        ));
    }
}

#[test]
fn rejects_wrong_layer_policies_including_the_final_full_layer() {
    for bytes in [BASE, MLX] {
        for index in [0, 1, 2, 5, 47, 48] {
            let mut config: serde_json::Value = serde_json::from_slice(bytes).unwrap();
            let old = config["layer_types"][index].as_str().unwrap();
            config["layer_types"][index] = if old == "full_attention" {
                "sliding_attention".into()
            } else {
                "full_attention".into()
            };
            assert_field(&config, "layer_types");
        }
        let mut config: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        config["layer_types"].as_array_mut().unwrap().pop();
        assert_field(&config, "layer_types");
        config.as_object_mut().unwrap().remove("layer_types");
        assert_field(&config, "layer_types");

        let mut config: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        config["no_rope_layer_indices"][13] = 47.into();
        assert_field(&config, "no_rope_layer_indices");
        let mut config: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        config["mlp_only_layers"] = serde_json::json!([0, 1, 2]);
        assert_field(&config, "mlp_only_layers");
    }
}

#[test]
fn rejects_conflicting_export_aliases_and_rope_overrides() {
    for (field, changed) in [
        ("dtype", serde_json::json!("float16")),
        ("scoring_func", serde_json::json!("softmax")),
        ("num_local_experts", serde_json::json!(128)),
        ("attention_sink", serde_json::json!(false)),
        ("router_expert_bias", serde_json::json!(true)),
    ] {
        let mut config: serde_json::Value = serde_json::from_slice(MLX).unwrap();
        config[field] = changed;
        assert_field(&config, field);
    }
    for (field, changed) in [
        ("rope_theta", serde_json::json!(20000)),
        ("rope_type", serde_json::json!("linear")),
    ] {
        let mut config: serde_json::Value = serde_json::from_slice(MLX).unwrap();
        config["rope_parameters"][field] = changed;
        assert_field(&config, "rope_parameters");
    }
    let mut config: serde_json::Value = serde_json::from_slice(BASE).unwrap();
    config["rope_scaling"] = serde_json::json!({"type": "linear", "factor": 2});
    assert_field(&config, "rope_scaling");
}

#[test]
fn rejects_unqualified_quantization_and_missing_embedding_override() {
    for (field, changed) in [
        ("bits", serde_json::json!(8)),
        ("group_size", serde_json::json!(32)),
        ("mode", serde_json::json!("mxfp4")),
    ] {
        let mut config: serde_json::Value = serde_json::from_slice(MLX).unwrap();
        config["quantization"][field] = changed;
        assert_field(&config, field);
    }
    for (field, changed) in [
        ("bits", serde_json::json!(8)),
        ("routed_expert_bits", serde_json::json!(8)),
        ("embedding_bits", serde_json::json!(4)),
    ] {
        let mut config: serde_json::Value = serde_json::from_slice(MLX).unwrap();
        config["mlx_export"][field] = changed;
        assert_field(&config, field);
    }
    let mut config: serde_json::Value = serde_json::from_slice(MLX).unwrap();
    config["mlx_export"]["full_attention_layers"][13] = 47.into();
    assert_field(&config, "full_attention_layers");

    let mut config: serde_json::Value = serde_json::from_slice(MLX).unwrap();
    config.as_object_mut().unwrap().remove("mlx_export");
    assert_field(&config, "mlx_export");
    let mut config: serde_json::Value = serde_json::from_slice(MLX).unwrap();
    config["quantization"] = false.into();
    assert_field(&config, "quantization");
    let mut config: serde_json::Value = serde_json::from_slice(MLX).unwrap();
    config["mlx_export"] = false.into();
    assert_field(&config, "mlx_export");
}

#[test]
fn export_kv_hints_do_not_admit_a_different_quantized_profile() {
    for bytes in [BASE, MLX] {
        for (field, changed) in [
            ("kv_key_bits", 4),
            ("kv_value_bits", 4),
            ("kv_group_size", 32),
        ] {
            let mut config: serde_json::Value = serde_json::from_slice(bytes).unwrap();
            config[field] = changed.into();
            assert_field(&config, field);
        }
    }
}

#[test]
fn rejects_malformed_policy_arrays_and_non_integer_token_ids() {
    for field in ["layer_types", "no_rope_layer_indices", "mlp_only_layers"] {
        for changed in [
            serde_json::Value::Null,
            serde_json::json!([]),
            serde_json::json!([false]),
        ] {
            let mut config: serde_json::Value = serde_json::from_slice(BASE).unwrap();
            config[field] = changed;
            assert_field(&config, field);
        }
    }
    for (field, changed) in [
        ("bos_token_id", serde_json::json!(150000.0)),
        ("pad_token_id", serde_json::json!(150001.0)),
        (
            "eos_token_id",
            serde_json::json!([150001.0, 150003, 150005, 150018]),
        ),
    ] {
        let mut config: serde_json::Value = serde_json::from_slice(BASE).unwrap();
        config[field] = changed;
        assert_field(&config, field);
    }
    let mut config: serde_json::Value = serde_json::from_slice(MLX).unwrap();
    config.as_object_mut().unwrap().remove("quantization");
    assert_field(&config, "quantization");
}

#[test]
#[cfg(not(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")))]
fn flint_prefix_reuse_opt_in_is_constant_false_off_the_mlx_target() {
    // Opus r1 SF-4: the snapshot store exists only on the MLX Mac target;
    // on every other build (CUDA servers, host test binaries) the env +
    // file flags must not arm a save that is a no-op. This test only
    // compiles where that must hold, so the day the cfg gate is deleted
    // The off-target half of the SF-4 pair (the policy lives in the
    // mlx_prefix_tests module doc): the cfg-false body, plus an isolated
    // CHILD run with the env actually set - the dangerous case - in the
    // self-spawn shape of mlx_batch_profile_native.rs.
    assert!(!super::prefix_reuse_opt_in());
    let binary = std::env::current_exe().expect("test binary");
    let output = std::process::Command::new(&binary)
        .args([
            "--exact",
            "family::flint::tests::flint_prefix_reuse_child_env_set_stays_off_mlx",
            "--test-threads",
            "1",
        ])
        .env("FIFI_FLINT_PREFIX_REUSE", "1")
        .output()
        .expect("spawn the isolated child test");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Opus wave-2 SF-3: exit status alone passes when --exact matches zero
    // tests, so the child must PROVE it ran exactly one test.
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "the child must run exactly one test with the env set and pass it (status {:?})",
        output.status.code()
    );
}

/// The retention-cap arithmetic is pure so the host suite pins it,
/// including the too-small-headroom refusal A's wave-2 ruling requires.
#[test]
fn flint_prefix_retention_cap_math_admits_and_fails_closed() {
    use super::{
        PREFIX_OS_FLOOR_BYTES, derived_retention_cap, retention_admitted, retention_cap_from_env,
    };
    // Derivation: physical - weights - floor, saturating; unknown
    // physical is None and must NEVER read as "no cap".
    assert_eq!(
        derived_retention_cap(Some(64 << 30), 4 << 30),
        Some((64 << 30) - (4 << 30) - PREFIX_OS_FLOOR_BYTES)
    );
    assert_eq!(derived_retention_cap(Some(8 << 30), 4 << 30), Some(0));
    assert_eq!(
        derived_retention_cap(None, 0),
        None,
        "unknown headroom fails closed"
    );

    // Admission: 2x new charge (entry + COW) + the retained overlap fits.
    let cap = (64 << 30) - (4 << 30) - PREFIX_OS_FLOOR_BYTES;
    assert!(retention_admitted(1 << 30, 0, cap));
    assert!(
        !retention_admitted(cap, 0, cap),
        "charge alone saturates the cap"
    );
    assert!(
        !retention_admitted(1 << 30, cap, cap),
        "retained overlap refuses"
    );
    // The faked too-small headroom (A ruling item 3): a 2 GiB cap cannot
    // admit a snapshot whose COW doubles past it.
    assert!(!retention_admitted((1 << 30) + 1, 0, 2 << 30));

    // Strict env parse: absent derives, unparsable or non-UTF-8 errs.
    use std::ffi::OsStr;
    assert_eq!(retention_cap_from_env(None), Ok(None));
    assert_eq!(
        retention_cap_from_env(Some(OsStr::new("4096"))),
        Ok(Some(4096))
    );
    assert!(retention_cap_from_env(Some(OsStr::new("1G"))).is_err());
    assert!(retention_cap_from_env(Some(OsStr::new("1e9"))).is_err());
    assert!(retention_cap_from_env(Some(OsStr::new("-1"))).is_err());
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let non_utf8 = OsStr::from_bytes(b"\xff\xfe");
        assert!(retention_cap_from_env(Some(non_utf8)).is_err());
    }
}

/// Opus wave-2 MF-1, the host half: the cap derivation is INERT unless
/// the reuse opt-in is armed - a default-OFF engine loads unchanged with
/// a garbage budget env, no PATH, and no physical-memory probe.
#[test]
fn flint_prefix_cap_is_inert_when_the_opt_in_is_off() {
    use super::retention_cap_for;
    // Proves the probes NEVER EXECUTE when the opt-in is off (lazy by
    // construction: FnOnce closures, not merely branched around).
    let probed = std::cell::Cell::new(false);
    assert_eq!(
        retention_cap_for(
            false,
            Some(std::ffi::OsStr::new("garbage-1G")),
            || {
                probed.set(true);
                None
            },
            || {
                probed.set(true);
                Err("walk".into())
            }
        ),
        Ok(0)
    );
    assert!(
        !probed.get(),
        "the probes must not run when the opt-in is off"
    );
    // Opt-in ON: strict env wins, non-UTF-8 and unparsable both error.
    assert_eq!(
        retention_cap_for(
            true,
            Some(std::ffi::OsStr::new("4096")),
            || None,
            || Err("walk".into())
        ),
        Ok(4096)
    );
    assert!(
        retention_cap_for(
            true,
            Some(std::ffi::OsStr::new("1G")),
            || None,
            || Err("walk".into())
        )
        .is_err()
    );
    // An explicit zero budget would refuse every save while prefill still
    // splits: rejected at startup, naming the var (Sol wave-3 SF-1).
    let zero = retention_cap_for(true, Some(std::ffi::OsStr::new("0")), || None, || Ok(0));
    let Err(text) = zero else {
        panic!("an explicit zero budget must be a startup error");
    };
    assert!(text.contains("FIFI_FLINT_PREFIX_BUDGET_BYTES"), "{text}");
    // Opt-in ON, env absent: derive; unknown physical fails closed.
    assert_eq!(
        retention_cap_for(true, None, || Some(64 << 30), || Ok(4 << 30)),
        Ok((64 << 30) - (4 << 30) - super::PREFIX_OS_FLOOR_BYTES)
    );
    assert!(retention_cap_for(true, None, || None, || Ok(0)).is_err());
    assert!(retention_cap_for(true, None, || Some(8 << 30), || Ok(4 << 30)).is_err());
}

/// The child half of the SF-4 coverage: a FRESH process with
/// FIFI_FLINT_PREFIX_REUSE=1, spawned by
/// [`flint_prefix_reuse_opt_in_is_constant_false_off_the_mlx_target`]. A
/// plain suite run (no env) returns at the guard; only the parent-driven
/// run makes the assertion load-bearing.
#[test]
fn flint_prefix_reuse_child_env_set_stays_off_mlx() {
    if std::env::var("FIFI_FLINT_PREFIX_REUSE").as_deref() != Ok("1") {
        return;
    }
    #[cfg(not(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")))]
    {
        // The cfg-false body must IGNORE the env (Sol wave-1 SF-4: this is
        // exactly the dangerous enabled-variable case).
        assert!(!super::prefix_reuse_opt_in());
    }
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    {
        // On the MLX target the env legitimately arms the opt-in; the
        // dangerous case is off-target builds, covered by the parent.
    }
}

/// One-off (owner ask, run-3ab follow-up): the tip boundary index of a
/// real served prompt. Loads ONLY the tokenizer (no model, no GPU),
/// renders one user message the way chat.rs turn 1 does, and prints
/// ids.len(), the derived tip, and the implied suffix length
/// (ids.len() - tip). Run with FLINT_QUAL_MODEL=<model dir> and
/// FLINT_QUAL_PROMPT=<prompt text file>; the run-3ab turn-1 prompt is
/// prefixwin3ab/prompt.txt on the Mac.
#[test]
#[ignore = "host-only one-off; real tokenizer; no GPU; run via FLINT_QUAL_MODEL + FLINT_QUAL_PROMPT"]
fn a_derived_tip_index_of_a_served_prompt() {
    let root = std::env::var_os("FLINT_QUAL_MODEL").map_or_else(
        || std::path::Path::new("/not-set").to_path_buf(),
        Into::into,
    );
    let prompt_path = std::env::var_os("FLINT_QUAL_PROMPT").map_or_else(
        || std::path::Path::new("/not-set").to_path_buf(),
        Into::into,
    );
    let text = std::fs::read_to_string(&prompt_path).expect("the prompt text file");
    let bundle = super::factory::load_tokenizer(&root).expect("load the real tokenizer");
    let msgs = vec![crate::tokenizer::PromptMessage {
        role: crate::tokenizer::Role::User,
        content: Some(text),
        reasoning_content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    }];
    let (ids, boundaries, _fallback) = bundle
        .encode_prompt_with_derived_boundary(&msgs, &[], crate::tokenizer::Thinking::Default)
        .expect("derive the boundary of a single-turn prompt");
    let tip = boundaries
        .tip
        .expect("a served single-turn prompt derives an internal tip");
    println!(
        "DERIVED-TIP ids={} tip={} suffix_len={}",
        ids.len(),
        tip,
        ids.len() - tip
    );
}
