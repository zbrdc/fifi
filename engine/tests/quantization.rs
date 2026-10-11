use inference_engine::{
    family::quantization::{ModuleQuantization, QuantizationMap},
    kernels::{QuantizationMode, QuantizationSpec},
};

#[test]
fn affine_defaults_resolve_after_input_bytes_drop() {
    let bytes = br#"{"quantization":{"mode":"affine","bits":4,"group_size":64}}"#.to_vec();
    let map = QuantizationMap::from_json(&bytes).unwrap();
    drop(bytes);
    assert_eq!(
        map.resolve("model.layers.0.mlp.up_proj").unwrap(),
        ModuleQuantization::Quantized(QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 4,
            group_size: 64,
        })
    );
}

#[test]
fn overrides_match_only_the_exact_case_sensitive_module() {
    let map = QuantizationMap::from_json(
        br#"{
        "quantization":{"mode":"affine","bits":4,"group_size":64,
            "model.layers.0.mlp.up_proj":{"mode":"affine","bits":6,"group_size":32}}
    }"#,
    )
    .unwrap();
    assert_eq!(
        map.resolve("model.layers.0.mlp.up_proj").unwrap(),
        ModuleQuantization::Quantized(QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 6,
            group_size: 32,
        })
    );
    for module in [
        "model.layers.0.mlp",
        "up_proj",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.UP_proj",
    ] {
        assert_eq!(
            map.resolve(module).unwrap(),
            ModuleQuantization::Quantized(QuantizationSpec {
                mode: QuantizationMode::Affine,
                bits: 4,
                group_size: 64,
            })
        );
    }
}

#[test]
fn literal_false_preserves_an_unquantized_override() {
    let map = QuantizationMap::from_json(
        br#"{
        "quantization":{"mode":"affine","bits":4,"group_size":64,"lm_head":false}
    }"#,
    )
    .unwrap();
    assert_eq!(
        map.resolve("lm_head").unwrap(),
        ModuleQuantization::Unquantized
    );
    assert_eq!(
        map.resolve("lm_head.weight").unwrap(),
        ModuleQuantization::Quantized(QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 4,
            group_size: 64,
        })
    );
}

#[test]
fn mlx_fallback_resolves_defaults_and_overrides() {
    let map = QuantizationMap::from_json(
        br#"{
        "quantization_config":{"mode":"affine","bits":5,"group_size":128,
            "lm_head":false,"model.proj":{"mode":"affine","bits":8,"group_size":32}}
    }"#,
    )
    .unwrap();
    assert_eq!(
        map.resolve("model.other").unwrap(),
        ModuleQuantization::Quantized(QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 5,
            group_size: 128,
        })
    );
    assert_eq!(
        map.resolve("model.proj").unwrap(),
        ModuleQuantization::Quantized(QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 8,
            group_size: 32,
        })
    );
    assert_eq!(
        map.resolve("lm_head").unwrap(),
        ModuleQuantization::Unquantized
    );
}

fn checkpoint_error(config: &[u8], context: &str) {
    let inference_engine::EngError::Checkpoint(message) =
        QuantizationMap::from_json(config).expect_err("malformed metadata must fail")
    else {
        panic!("expected a checkpoint error");
    };
    assert!(message.contains(context), "{message}");
}

#[test]
fn both_metadata_keys_must_validate_and_agree_regardless_of_order() {
    let primary = r#"{"mode":"affine","bits":4,"group_size":64,"head":false,"proj":{"mode":"affine","bits":6,"group_size":32}}"#;
    let equal = r#"{"proj":{"group_size":32,"bits":6,"mode":"affine"},"head":false,"group_size":64,"bits":4,"mode":"affine"}"#;
    let map = QuantizationMap::from_json(
        format!(r#"{{"quantization":{primary},"quantization_config":{equal}}}"#).as_bytes(),
    )
    .unwrap();
    assert_eq!(
        map.resolve("proj").unwrap(),
        ModuleQuantization::Quantized(QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 6,
            group_size: 32,
        })
    );
    for conflicting in [
        primary.replace(r#""bits":4"#, r#""bits":8"#),
        primary.replace(
            r#""head":false"#,
            r#""head":{"mode":"affine","bits":4,"group_size":64}"#,
        ),
        primary.replace(r#""head":false,"#, ""),
        "null".into(),
        r#"{"mode":"affine","bits":4}"#.into(),
    ] {
        checkpoint_error(
            format!(r#"{{"quantization":{primary},"quantization_config":{conflicting}}}"#)
                .as_bytes(),
            "quantization_config",
        );
        checkpoint_error(
            format!(r#"{{"quantization_config":{primary},"quantization":{conflicting}}}"#)
                .as_bytes(),
            "quantization",
        );
    }
}

#[test]
fn affine_settings_accept_only_the_declared_widths_and_groups() {
    for bits in [2, 3, 4, 5, 6, 8] {
        for group_size in [32, 64, 128] {
            let config = format!(
                r#"{{"quantization":{{"mode":"affine","bits":{bits},"group_size":{group_size}}}}}"#
            );
            let map = QuantizationMap::from_json(config.as_bytes()).unwrap();
            assert_eq!(
                map.resolve("proj").unwrap(),
                ModuleQuantization::Quantized(QuantizationSpec {
                    mode: QuantizationMode::Affine,
                    bits,
                    group_size,
                })
            );
        }
    }
    for (bits, group_size, field) in [
        (0, 64, "bits"),
        (1, 64, "bits"),
        (7, 64, "bits"),
        (16, 64, "bits"),
        (4, 0, "group_size"),
        (4, 16, "group_size"),
        (4, 256, "group_size"),
    ] {
        checkpoint_error(
            format!(
                r#"{{"quantization":{{"mode":"affine","bits":{bits},"group_size":{group_size}}}}}"#
            )
            .as_bytes(),
            field,
        );
    }
}

#[test]
fn mxfp_metadata_requires_its_exact_bits_and_group_size() {
    let map = QuantizationMap::from_json(
        br#"{
        "quantization":{"mode":"mxfp4","bits":4,"group_size":32,
            "proj":{"mode":"mxfp8","bits":8,"group_size":32}}
    }"#,
    )
    .unwrap();
    assert_eq!(
        map.resolve("other").unwrap(),
        ModuleQuantization::Quantized(QuantizationSpec {
            mode: QuantizationMode::Mxfp4,
            bits: 4,
            group_size: 32,
        })
    );
    assert_eq!(
        map.resolve("proj").unwrap(),
        ModuleQuantization::Quantized(QuantizationSpec {
            mode: QuantizationMode::Mxfp8,
            bits: 8,
            group_size: 32,
        })
    );
    for (mode, bits, group_size, field) in [
        ("mxfp4", 8, 32, "bits"),
        ("mxfp8", 4, 32, "bits"),
        ("mxfp4", 4, 64, "group_size"),
        ("mxfp8", 8, 128, "group_size"),
    ] {
        checkpoint_error(
            format!(
                r#"{{"quantization":{{"mode":"{mode}","bits":{bits},"group_size":{group_size}}}}}"#
            )
            .as_bytes(),
            field,
        );
    }
}

#[test]
fn module_paths_are_validated_in_overrides_and_lookups() {
    let map = QuantizationMap::from_json(
        br#"{"quantization":{"mode":"affine","bits":4,"group_size":64}}"#,
    )
    .unwrap();
    for module in [
        "",
        ".proj",
        "proj.",
        "model..proj",
        "model/proj",
        " model",
        "model x",
        "model\nproj",
        "model-proj",
        "模型",
        "proj\\path",
    ] {
        let inference_engine::EngError::Checkpoint(message) = map
            .resolve(module)
            .expect_err("invalid lookup path must fail")
        else {
            panic!("expected a checkpoint error");
        };
        assert!(message.contains("module"), "{message}");
        let config = serde_json::json!({"quantization":{"mode":"affine","bits":4,"group_size":64, module: false}});
        checkpoint_error(&serde_json::to_vec(&config).unwrap(), "module");
    }
    assert_eq!(
        map.resolve("_model.Layer_0.012.proj").unwrap(),
        ModuleQuantization::Quantized(QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 4,
            group_size: 64,
        })
    );
}

#[test]
fn overrides_require_false_or_valid_settings_without_extra_fields() {
    for setting in [
        r#"{"mode":"affine","bits":4,"group_size":64,"extra":false}"#,
        "true",
        "null",
        "[]",
        "4",
        r#""affine""#,
        "{}",
        r#"{"mode":"affine","bits":4}"#,
        r#"{"mode":null,"bits":4,"group_size":64}"#,
        r#"{"mode":"affine","group_size":64}"#,
        r#"{"mode":"unknown","bits":4,"group_size":64}"#,
        r#"{"mode":"affine","bits":7,"group_size":64}"#,
    ] {
        checkpoint_error(format!(r#"{{"quantization":{{"mode":"affine","bits":4,"group_size":64,"model.proj":{setting}}}}}"#).as_bytes(), "model.proj");
    }
}

#[test]
fn fallback_rejects_source_format_markers_even_with_mlx_defaults() {
    for quant_method in [
        r#""awq""#,
        "false",
        r#"{"mode":"affine","bits":4,"group_size":64}"#,
    ] {
        let foreign = format!(
            r#"{{"mode":"affine","bits":4,"group_size":64,"quant_method":{quant_method}}}"#
        );
        for key in ["quantization", "quantization_config"] {
            checkpoint_error(
                format!(r#"{{"{key}":{foreign}}}"#).as_bytes(),
                "quant_method",
            );
        }
    }
}

#[test]
fn duplicate_json_members_fail_at_every_depth_before_values_are_lost() {
    let defaults = r#""quantization":{"mode":"affine","bits":4,"group_size":64}"#;
    for config in [
        format!("{{{defaults},{defaults}}}"),
        r#"{"quantization":{"mode":"affine","bits":8,"bits":4,"group_size":64}}"#.into(),
        r#"{"quantization":{"mode":"affine","bits":4,"group_size":64,"head":false,"head":false}}"#.into(),
        r#"{"quantization":{"mode":"affine","bits":4,"group_size":64,"head":{"mode":"affine","bits":4,"group_size":32,"group_size":64}}}"#.into(),
        r#"{"quantization":{"mode":"affine","bits":4,"b\u0069ts":4,"group_size":64}}"#.into(),
        format!(r#"{{{defaults},"unrelated":{{"a":0,"a":1}}}}"#),
        format!(r#"{{{defaults},"unrelated":[{{"a":0,"a":1}}]}}"#),
    ] {
        checkpoint_error(config.as_bytes(), "duplicate");
    }
}

#[test]
fn metadata_numbers_must_be_u32_json_integers() {
    for value in [
        "-1",
        "-0",
        "4.0",
        "4e0",
        r#""4""#,
        "true",
        "null",
        "[]",
        "{}",
        "4294967296",
        "18446744073709551616",
    ] {
        checkpoint_error(
            format!(r#"{{"quantization":{{"mode":"affine","bits":{value},"group_size":64}}}}"#)
                .as_bytes(),
            "bits",
        );
        checkpoint_error(
            format!(r#"{{"quantization":{{"mode":"affine","bits":4,"group_size":{value}}}}}"#)
                .as_bytes(),
            "group_size",
        );
    }
}

#[test]
fn malformed_missing_and_excessively_nested_configurations_fail() {
    for config in [
        "",
        "{",
        "{}",
        "[]",
        "null",
        "false",
        "4",
        r#""config""#,
        r#"{"quantization":null}"#,
        r#"{"quantization":false}"#,
        r#"{"quantization":[]}"#,
        r#"{"quantization":{"bits":4,"group_size":64}}"#,
        r#"{"quantization":{"mode":"affine","group_size":64}}"#,
        r#"{"quantization":{"mode":"affine","bits":4}}"#,
        r#"{"quantization":{"mode":"Affine","bits":4,"group_size":64}}"#,
        r#"{"quantization":{"mode":4,"bits":4,"group_size":64}}"#,
        r#"{"quantization":{"mode":"affine","bits":4,"group_size":64}} false"#,
    ] {
        assert!(
            matches!(
                QuantizationMap::from_json(config.as_bytes()),
                Err(inference_engine::EngError::Checkpoint(_))
            ),
            "{config}"
        );
    }
    checkpoint_error(b"{\"quantization\":\xff}", "JSON");
    let nested = format!(
        r#"{{"quantization":{{"mode":"affine","bits":4,"group_size":64}},"other":{}0{}}}"#,
        "[".repeat(256),
        "]".repeat(256)
    );
    checkpoint_error(nested.as_bytes(), "recursion limit");
}

#[test]
fn unrelated_model_fields_do_not_change_resolved_settings() {
    let map = QuantizationMap::from_json(
        br#"{
        "quantization":{"mode":"affine","bits":4,"group_size":64},
        "model_type":"example", "escaped":"a\nb", "enabled":true, "unused":null,
        "layers":12, "offset":-1, "temperature":0.25,
        "nested":[{"same":1},{"same":2},[false,null,"word"]]
    }"#,
    )
    .unwrap();
    assert_eq!(
        map.resolve("model.proj").unwrap(),
        ModuleQuantization::Quantized(QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 4,
            group_size: 64,
        })
    );
}

#[test]
fn pinned_flash_next_settings_preserve_mixed_widths_and_groups() {
    let map = QuantizationMap::from_json(include_bytes!("fixtures/oq-metadata.json")).unwrap();
    for (module, bits, group_size) in [
        (
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0",
            4,
            32,
        ),
        (
            "language_model.model.layers.0.attn_hyper_connection.block_inject_weight",
            5,
            64,
        ),
        ("language_model.model.layers.0.linear_attn.in_proj_a", 6, 64),
        (
            "language_model.model.layers.0.mlp.shared_expert.down_proj",
            8,
            128,
        ),
    ] {
        assert_eq!(
            map.resolve(module).unwrap(),
            ModuleQuantization::Quantized(QuantizationSpec {
                mode: QuantizationMode::Affine,
                bits,
                group_size,
            })
        );
    }
}

#[test]
fn legacy_override_mode_defaults_to_affine_without_inheriting_global_mode() {
    for key in ["quantization", "quantization_config"] {
        let config = format!(
            r#"{{"{key}":{{"mode":"mxfp4","bits":4,"group_size":32,"gate":{{"bits":8,"group_size":64}}}}}}"#
        );
        let map = QuantizationMap::from_json(config.as_bytes()).unwrap();
        assert_eq!(
            map.resolve("gate").unwrap(),
            ModuleQuantization::Quantized(QuantizationSpec {
                mode: QuantizationMode::Affine,
                bits: 8,
                group_size: 64,
            })
        );
        assert_eq!(
            map.resolve("other").unwrap(),
            ModuleQuantization::Quantized(QuantizationSpec {
                mode: QuantizationMode::Mxfp4,
                bits: 4,
                group_size: 32,
            })
        );
    }
}
