use inference_engine::family::nemotron35::{
    NemotronActivation, NemotronBlockKind, NemotronTextSpec,
};

const CONFIG: &[u8] = include_bytes!("fixtures/nemotron35/config.json");
const PINNED_BLOCK_ORDER: &str =
    "memememaememememaememememaemememememaemememememaemememememaemememememaememememaememememe";

fn pinned_config() -> Vec<u8> {
    std::env::var_os("NEM35_COMPLETE_CONFIG")
        .map(|path| std::fs::read(path).expect("complete pinned configuration"))
        .unwrap_or_else(|| CONFIG.to_vec())
}

#[test]
fn accepts_pinned_nemotron_text_config_and_preserves_order_and_geometry() {
    let config = pinned_config();
    let spec = NemotronTextSpec::from_config(&config).expect("pinned architecture config");
    let order: String = spec
        .block_order()
        .iter()
        .map(|kind| match kind {
            NemotronBlockKind::Mamba => 'm',
            NemotronBlockKind::Moe => 'e',
            NemotronBlockKind::Attention => 'a',
        })
        .collect();
    assert_eq!(order, PINNED_BLOCK_ORDER);
    assert_eq!(order.len(), 88);
    assert_eq!(order.matches('m').count(), 40);
    assert_eq!(order.matches('e').count(), 40);
    assert_eq!(order.matches('a').count(), 8);

    let geometry = spec.geometry();
    assert_eq!(geometry.hidden_size, 4096);
    assert_eq!(geometry.vocab_size, 131_072);
    assert_eq!(geometry.num_attention_heads, 32);
    assert_eq!(geometry.num_key_value_heads, 2);
    assert_eq!(geometry.head_dim, 128);
    assert_eq!(geometry.max_position_embeddings, 1_048_576);
    assert_eq!(geometry.layer_norm_epsilon, 1e-5);

    let mamba = spec.mamba_geometry();
    assert_eq!(mamba.state_size, 128);
    assert_eq!(mamba.conv_kernel, 4);
    assert_eq!(mamba.heads, 128);
    assert_eq!(mamba.head_dim, 64);
    assert_eq!(mamba.expand, 2);
    assert_eq!(mamba.inner_size, 8192);
    assert_eq!(mamba.chunk_size, 128);

    let moe = spec.moe_geometry();
    assert_eq!(moe.intermediate_size, 2688);
    assert_eq!(moe.moe_intermediate_size, 2688);
    assert_eq!(moe.latent_size, 1024);
    assert_eq!(moe.shared_expert_intermediate_size, 5376);
    assert_eq!(moe.routed_experts, 512);
    assert_eq!(moe.shared_experts, 1);
    assert_eq!(moe.experts_per_token, 22);
    assert_eq!(moe.n_group, 1);
    assert_eq!(moe.n_groups, 8);
    assert_eq!(moe.topk_group, 1);
    assert!(moe.norm_topk_prob);
    assert_eq!(moe.routed_scaling_factor, 5.0);

    let activations = spec.activations();
    assert_eq!(activations.mamba, NemotronActivation::Silu);
    assert_eq!(activations.mlp, NemotronActivation::Relu2);
}

#[test]
fn rejects_wrong_wrapper_family() {
    use inference_engine::EngError;

    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    config["model_type"] = serde_json::json!("other_family");
    let changed = serde_json::to_vec(&config).expect("serialized config");
    match NemotronTextSpec::from_config(&changed) {
        Err(EngError::Checkpoint(message)) => assert!(message.contains("model_type")),
        other => panic!("expected wrong-family checkpoint error, got {other:?}"),
    }
}

#[test]
fn rejects_wrong_text_model_type() {
    use inference_engine::EngError;

    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    config["llm_config"]["model_type"] = serde_json::json!("other_text_model");
    let changed = serde_json::to_vec(&config).expect("serialized config");
    match NemotronTextSpec::from_config(&changed) {
        Err(EngError::Checkpoint(message)) => assert!(message.contains("model_type")),
        other => panic!("expected text-family checkpoint error, got {other:?}"),
    }
}

#[test]
fn rejects_missing_wrapper_architecture() {
    use inference_engine::EngError;

    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    config
        .as_object_mut()
        .expect("root object")
        .remove("architectures");
    let changed = serde_json::to_vec(&config).expect("serialized config");
    match NemotronTextSpec::from_config(&changed) {
        Err(EngError::Checkpoint(message)) => assert!(message.contains("architectures")),
        other => panic!("expected missing-architecture checkpoint error, got {other:?}"),
    }
}

#[test]
fn rejects_missing_text_architecture() {
    use inference_engine::EngError;

    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    config["llm_config"]
        .as_object_mut()
        .expect("text object")
        .remove("architectures");
    let changed = serde_json::to_vec(&config).expect("serialized config");
    match NemotronTextSpec::from_config(&changed) {
        Err(EngError::Checkpoint(message)) => assert!(message.contains("architectures")),
        other => panic!("expected missing-text-architecture error, got {other:?}"),
    }
}

#[test]
fn rejects_reordered_mamba_and_moe_blocks() {
    use inference_engine::EngError;

    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    config["llm_config"]["layers_block_type"][0] = serde_json::json!("moe");
    config["llm_config"]["layers_block_type"][1] = serde_json::json!("mamba");
    let changed = serde_json::to_vec(&config).expect("serialized config");
    match NemotronTextSpec::from_config(&changed) {
        Err(EngError::Checkpoint(message)) => assert!(message.contains("layers_block_type")),
        other => panic!("expected block-order checkpoint error, got {other:?}"),
    }
}

#[test]
fn rejects_unsupported_hidden_geometry() {
    use inference_engine::EngError;

    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    config["llm_config"]["hidden_size"] = serde_json::json!(2048);
    let changed = serde_json::to_vec(&config).expect("serialized config");
    let result = NemotronTextSpec::from_config(&changed);
    assert!(
        matches!(result, Err(EngError::Checkpoint(ref message)) if message.contains("hidden_size")),
        "unsupported hidden_size was accepted or misreported: {result:?}"
    );
}

#[test]
fn rejects_unsupported_mamba_convolution_geometry() {
    use inference_engine::EngError;

    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    config["llm_config"]["conv_kernel"] = serde_json::json!(3);
    let changed = serde_json::to_vec(&config).expect("serialized config");
    let result = NemotronTextSpec::from_config(&changed);
    assert!(
        matches!(result, Err(EngError::Checkpoint(ref message)) if message.contains("conv_kernel")),
        "unsupported conv_kernel was accepted or misreported: {result:?}"
    );
}

#[test]
fn rejects_unsupported_grouped_router_geometry() {
    use inference_engine::EngError;

    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    config["llm_config"]["n_groups"] = serde_json::json!(16);
    let changed = serde_json::to_vec(&config).expect("serialized config");
    let result = NemotronTextSpec::from_config(&changed);
    assert!(
        matches!(result, Err(EngError::Checkpoint(ref message)) if message.contains("n_groups")),
        "unsupported n_groups was accepted or misreported: {result:?}"
    );
}

#[test]
fn rejects_mamba_activation_outside_pinned_declaration() {
    use inference_engine::EngError;

    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    config["llm_config"]["mamba_hidden_act"] = serde_json::json!("relu2");
    let changed = serde_json::to_vec(&config).expect("serialized config");
    let result = NemotronTextSpec::from_config(&changed);
    assert!(
        matches!(result, Err(EngError::Checkpoint(ref message)) if message.contains("mamba_hidden_act")),
        "unsupported Mamba activation was accepted or misreported: {result:?}"
    );
}

#[test]
fn rejects_mlp_activation_outside_pinned_declaration() {
    use inference_engine::EngError;

    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    config["llm_config"]["mlp_hidden_act"] = serde_json::json!("silu");
    let changed = serde_json::to_vec(&config).expect("serialized config");
    let result = NemotronTextSpec::from_config(&changed);
    assert!(
        matches!(result, Err(EngError::Checkpoint(ref message)) if message.contains("mlp_hidden_act")),
        "unsupported MLP activation was accepted or misreported: {result:?}"
    );
}

#[test]
fn rejects_root_and_nested_duplicate_json_members() {
    use inference_engine::EngError;

    let original = std::str::from_utf8(CONFIG).expect("fixture UTF-8");
    let root_duplicate = original.replacen(
        "\"model_type\": \"nemotron_h_omni\",",
        "\"model_type\": \"other_family\", \"model_type\": \"nemotron_h_omni\",",
        1,
    );
    let nested_duplicate = original.replacen(
        "\"hidden_size\": 4096,",
        "\"hidden_size\": 4095, \"hidden_size\": 4096,",
        1,
    );
    let root_result = NemotronTextSpec::from_config(root_duplicate.as_bytes());
    let nested_result = NemotronTextSpec::from_config(nested_duplicate.as_bytes());
    let root_rejected = matches!(
        root_result,
        Err(EngError::Checkpoint(message))
            if message.contains("duplicate JSON member") && message.contains("model_type")
    );
    let nested_rejected = matches!(
        nested_result,
        Err(EngError::Checkpoint(message))
            if message.contains("duplicate JSON member") && message.contains("hidden_size")
    );
    assert!(root_rejected, "duplicate root JSON member was accepted");
    assert!(nested_rejected, "duplicate nested JSON member was accepted");
}

#[test]
fn rejects_malformed_json_types_and_missing_required_fields() {
    use inference_engine::EngError;

    assert!(matches!(
        NemotronTextSpec::from_config(b"{"),
        Err(EngError::Checkpoint(message)) if message.contains("JSON")
    ));

    let mut malformed: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    malformed["llm_config"]["hidden_size"] = serde_json::json!("4096");
    let malformed = serde_json::to_vec(&malformed).expect("serialized config");
    assert!(matches!(
        NemotronTextSpec::from_config(&malformed),
        Err(EngError::Checkpoint(message)) if message.contains("hidden_size")
    ));

    let mut missing: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    let removed = missing["llm_config"]
        .as_object_mut()
        .expect("text config object")
        .remove("ssm_state_size");
    assert!(removed.is_some());
    let missing = serde_json::to_vec(&missing).expect("serialized config");
    assert!(matches!(
        NemotronTextSpec::from_config(&missing),
        Err(EngError::Checkpoint(message)) if message.contains("ssm_state_size")
    ));
}

#[test]
fn rejects_unsupported_block_kind() {
    use inference_engine::EngError;

    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    config["llm_config"]["layers_block_type"][22] = serde_json::json!("linear_attention");
    let changed = serde_json::to_vec(&config).expect("serialized config");
    assert!(matches!(
        NemotronTextSpec::from_config(&changed),
        Err(EngError::Checkpoint(message)) if message.contains("layers_block_type[22]")
    ));
}

#[test]
fn rejects_unrepresentable_and_overflowing_numeric_extents() {
    use inference_engine::EngError;

    let mut out_of_range: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    out_of_range["llm_config"]["hidden_size"] = serde_json::json!(4_294_967_296_u64);
    let out_of_range = serde_json::to_vec(&out_of_range).expect("serialized config");
    assert!(matches!(
        NemotronTextSpec::from_config(&out_of_range),
        Err(EngError::Checkpoint(message))
            if message.contains("hidden_size") && message.contains("u32 integer")
    ));

    let mut attention: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    attention["llm_config"]["num_attention_heads"] = serde_json::json!(u32::MAX);
    let attention = serde_json::to_vec(&attention).expect("serialized config");
    assert!(matches!(
        NemotronTextSpec::from_config(&attention),
        Err(EngError::Checkpoint(message))
            if message.contains("attention projection extent") && message.contains("overflows u32")
    ));

    let mut mamba: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    mamba["llm_config"]["expand"] = serde_json::json!(u32::MAX);
    let mamba = serde_json::to_vec(&mamba).expect("serialized config");
    assert!(matches!(
        NemotronTextSpec::from_config(&mamba),
        Err(EngError::Checkpoint(message))
            if message.contains("Mamba expanded hidden extent") && message.contains("overflows u32")
    ));

    let mut experts: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    experts["llm_config"]["n_routed_experts"] = serde_json::json!(4_294_967_288_u32);
    let experts = serde_json::to_vec(&experts).expect("serialized config");
    assert!(matches!(
        NemotronTextSpec::from_config(&experts),
        Err(EngError::Checkpoint(message))
            if message.contains("routed expert extent") && message.contains("overflows u32")
    ));
}

#[test]
fn rejects_unsupported_normalization_and_top_k_geometry() {
    use inference_engine::EngError;

    let mut normalization: serde_json::Value =
        serde_json::from_slice(CONFIG).expect("fixture JSON");
    normalization["llm_config"]["layer_norm_epsilon"] = serde_json::json!(0.0);
    let normalization = serde_json::to_vec(&normalization).expect("serialized config");
    assert!(matches!(
        NemotronTextSpec::from_config(&normalization),
        Err(EngError::Checkpoint(message)) if message.contains("layer_norm_epsilon")
    ));

    let mut top_k: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    top_k["llm_config"]["num_experts_per_tok"] = serde_json::json!(23);
    let top_k = serde_json::to_vec(&top_k).expect("serialized config");
    assert!(matches!(
        NemotronTextSpec::from_config(&top_k),
        Err(EngError::Checkpoint(message)) if message.contains("num_experts_per_tok")
    ));

    let mut latent: serde_json::Value = serde_json::from_slice(CONFIG).expect("fixture JSON");
    latent["llm_config"]["moe_latent_size"] = serde_json::json!(512);
    let latent = serde_json::to_vec(&latent).expect("serialized config");
    assert!(matches!(
        NemotronTextSpec::from_config(&latent),
        Err(EngError::Checkpoint(message)) if message.contains("moe_latent_size")
    ));
}

#[test]
fn text_plan_carries_mamba_groups_and_residual_and_embedding_policy() {
    let spec = NemotronTextSpec::from_config(CONFIG).expect("pinned text config");
    assert_eq!(spec.mamba_geometry().groups, 8);
    assert!(!spec.tie_word_embeddings());
    assert!(spec.rescale_prenorm_residual());
    for (member, wrong) in [
        ("tie_word_embeddings", true),
        ("rescale_prenorm_residual", false),
    ] {
        let mut config: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
        config["llm_config"][member] = wrong.into();
        let error = NemotronTextSpec::from_config(&serde_json::to_vec(&config).unwrap())
            .expect_err("unsupported text policy");
        assert!(
            matches!(error, inference_engine::EngError::Checkpoint(message) if message.contains(member))
        );
    }
}

#[test]
fn mtp_metadata_is_recorded_but_rejected_from_execution() {
    let spec = NemotronTextSpec::from_config(CONFIG).unwrap();
    for member in [
        "llm_config.num_nextn_predict_layers",
        "llm_config.mtp_layers_block_type",
    ] {
        assert!(spec.excluded_members().iter().any(|name| name == member));
        assert!(matches!(spec.validate_execution_member(member),
            Err(inference_engine::EngError::Checkpoint(message))
                if message.contains(member) && message.contains("out of scope")));
    }
    assert!(
        spec.validate_execution_member("llm_config.hidden_size")
            .is_ok()
    );
}

#[test]
fn vision_metadata_is_recorded_but_rejected_from_execution() {
    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    config["vision_config"] = serde_json::json!({"hidden_size": 1536});
    config["video_context_token"] = serde_json::json!("<video>");
    config["projector_hidden_size"] = serde_json::json!(4096);
    let spec = NemotronTextSpec::from_config(&serde_json::to_vec(&config).unwrap()).unwrap();
    for member in [
        "vision_config",
        "video_context_token",
        "projector_hidden_size",
    ] {
        assert!(spec.excluded_members().iter().any(|name| name == member));
        assert!(matches!(spec.validate_execution_member(member),
            Err(inference_engine::EngError::Checkpoint(message)) if message.contains(member)));
    }
    assert!(
        spec.validate_execution_member("vision_config.hidden_size")
            .is_err()
    );
}

#[test]
fn audio_metadata_is_recorded_but_rejected_from_execution() {
    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    config["sound_config"] = serde_json::json!({"hidden_size": 1024});
    config["sound_context_token"] = serde_json::json!("<sound>");
    config["audio_config"] = serde_json::json!({});
    let spec = NemotronTextSpec::from_config(&serde_json::to_vec(&config).unwrap()).unwrap();
    for member in ["sound_config", "sound_context_token", "audio_config"] {
        assert!(spec.excluded_members().iter().any(|name| name == member));
        assert!(matches!(spec.validate_execution_member(member),
            Err(inference_engine::EngError::Checkpoint(message)) if message.contains(member)));
    }
}

#[test]
fn attention_has_explicit_no_rotation_with_reference_provenance() {
    use inference_engine::family::nemotron35::{NEMOTRON_ROTATION_PROVENANCE, NemotronRotation};
    let config: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    assert!(
        config["llm_config"]
            .as_object()
            .unwrap()
            .keys()
            .all(|name| !name.contains("rope") && !name.contains("theta"))
    );
    let spec = NemotronTextSpec::from_config(CONFIG).unwrap();
    assert_eq!(spec.rotation(), NemotronRotation::None);
    assert!(NEMOTRON_ROTATION_PROVENANCE.contains("transformers 5.10.0.dev0"));
    assert!(NEMOTRON_ROTATION_PROVENANCE.contains("NemotronHAttention.forward"));
    assert!(
        NEMOTRON_ROTATION_PROVENANCE
            .contains("859ff78fae71fa3f8857e4ea0620c43aebea624e310e8422c124c507131398d2")
    );
    spec.validate_execution_readiness()
        .expect("text attention needs no rotary constants");
}

#[test]
fn key_plan_maps_pinned_text_names_to_block_roles() {
    use inference_engine::family::nemotron35::{NemotronKeyClass as K, NemotronTensorPart as P};
    let spec = NemotronTextSpec::from_config(CONFIG).unwrap();
    for (name, block, class, expert, part) in [
        (
            "language_model.backbone.embeddings.weight",
            None,
            K::Embedding,
            None,
            P::Weight,
        ),
        (
            "language_model.backbone.norm_f.weight",
            None,
            K::FinalNorm,
            None,
            P::Weight,
        ),
        (
            "language_model.lm_head.weight_scale",
            None,
            K::LmHead,
            None,
            P::WeightScale,
        ),
        (
            "language_model.backbone.layers.0.norm.weight",
            Some(0),
            K::BlockNorm,
            None,
            P::Weight,
        ),
        (
            "language_model.backbone.layers.0.mixer.A_log",
            Some(0),
            K::MambaStateDecay,
            None,
            P::Value,
        ),
        (
            "language_model.backbone.layers.0.mixer.D",
            Some(0),
            K::MambaSkip,
            None,
            P::Value,
        ),
        (
            "language_model.backbone.layers.0.mixer.dt_bias",
            Some(0),
            K::MambaTimeBias,
            None,
            P::Value,
        ),
        (
            "language_model.backbone.layers.0.mixer.conv1d.bias",
            Some(0),
            K::MambaConvolution,
            None,
            P::Bias,
        ),
        (
            "language_model.backbone.layers.0.mixer.in_proj.input_scale",
            Some(0),
            K::MambaInputProjection,
            None,
            P::InputScale,
        ),
        (
            "language_model.backbone.layers.0.mixer.norm.weight",
            Some(0),
            K::MambaNorm,
            None,
            P::Weight,
        ),
        (
            "language_model.backbone.layers.0.mixer.out_proj.weight",
            Some(0),
            K::MambaOutputProjection,
            None,
            P::Weight,
        ),
        (
            "language_model.backbone.layers.7.mixer.q_proj.weight",
            Some(7),
            K::AttentionQuery,
            None,
            P::Weight,
        ),
        (
            "language_model.backbone.layers.7.mixer.k_proj.weight",
            Some(7),
            K::AttentionKey,
            None,
            P::Weight,
        ),
        (
            "language_model.backbone.layers.7.mixer.v_proj.weight",
            Some(7),
            K::AttentionValue,
            None,
            P::Weight,
        ),
        (
            "language_model.backbone.layers.7.mixer.o_proj.weight",
            Some(7),
            K::AttentionOutput,
            None,
            P::Weight,
        ),
        (
            "language_model.backbone.layers.1.mixer.gate.weight",
            Some(1),
            K::MoeRouter,
            None,
            P::Weight,
        ),
        (
            "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias",
            Some(1),
            K::MoeRouterCorrection,
            None,
            P::Value,
        ),
        (
            "language_model.backbone.layers.1.mixer.fc1_latent_proj.weight",
            Some(1),
            K::MoeLatentInput,
            None,
            P::Weight,
        ),
        (
            "language_model.backbone.layers.1.mixer.fc2_latent_proj.weight",
            Some(1),
            K::MoeLatentOutput,
            None,
            P::Weight,
        ),
        (
            "language_model.backbone.layers.1.mixer.experts.511.up_proj.weight_scale_2",
            Some(1),
            K::MoeExpertUp,
            Some(511),
            P::WeightScale2,
        ),
        (
            "language_model.backbone.layers.1.mixer.experts.0.down_proj.weight",
            Some(1),
            K::MoeExpertDown,
            Some(0),
            P::Weight,
        ),
        (
            "language_model.backbone.layers.1.mixer.shared_experts.up_proj.weight",
            Some(1),
            K::MoeSharedUp,
            None,
            P::Weight,
        ),
        (
            "language_model.backbone.layers.1.mixer.shared_experts.down_proj.weight",
            Some(1),
            K::MoeSharedDown,
            None,
            P::Weight,
        ),
    ] {
        let key = spec.key_plan(name).unwrap();
        assert_eq!(
            (key.block, key.class, key.expert, key.part),
            (block, class, expert, part),
            "{name}"
        );
    }
    for (kind, expected) in [
        (
            NemotronBlockKind::Mamba,
            vec![
                K::BlockNorm,
                K::MambaStateDecay,
                K::MambaSkip,
                K::MambaTimeBias,
                K::MambaConvolution,
                K::MambaInputProjection,
                K::MambaNorm,
                K::MambaOutputProjection,
            ],
        ),
        (
            NemotronBlockKind::Attention,
            vec![
                K::BlockNorm,
                K::AttentionQuery,
                K::AttentionKey,
                K::AttentionValue,
                K::AttentionOutput,
            ],
        ),
        (
            NemotronBlockKind::Moe,
            vec![
                K::BlockNorm,
                K::MoeRouter,
                K::MoeRouterCorrection,
                K::MoeLatentInput,
                K::MoeLatentOutput,
                K::MoeExpertUp,
                K::MoeExpertDown,
                K::MoeSharedUp,
                K::MoeSharedDown,
            ],
        ),
    ] {
        assert_eq!(kind.expected_key_classes(), expected);
    }
}

#[test]
fn key_plan_rejects_mtp_vision_audio_execution_with_named_members() {
    let spec = NemotronTextSpec::from_config(CONFIG).unwrap();
    for (name, member) in [
        (
            "language_model.mtp.layers.0.norm.weight",
            "num_nextn_predict_layers",
        ),
        ("vision_model.radio_model.weight", "vision_config"),
        (
            "vision_projector.vision_final_layernorm.weight",
            "vision_config",
        ),
        ("mlp1.0.weight", "projector_hidden_size"),
        ("sound_model.weight", "sound_config"),
        ("audio_model.weight", "audio_config"),
    ] {
        assert!(
            matches!(spec.key_plan(name),
            Err(inference_engine::EngError::Checkpoint(message))
                if message.contains(member) && message.contains("out of scope")),
            "{name}"
        );
    }
}

#[test]
fn key_plan_rejects_wrong_block_types_extents_parts_and_unknown_names() {
    let spec = NemotronTextSpec::from_config(CONFIG).unwrap();
    for name in [
        "language_model.backbone.layers.7.mixer.A_log",
        "language_model.backbone.layers.0.mixer.q_proj.weight",
        "language_model.backbone.layers.1.mixer.norm.weight",
        "language_model.backbone.layers.88.norm.weight",
        "language_model.backbone.layers.429496729600000000000.norm.weight",
        "language_model.backbone.layers.-1.norm.weight",
        "language_model.backbone.layers.1.mixer.experts.512.up_proj.weight",
        "language_model.backbone.layers.1.mixer.experts.x.up_proj.weight",
        "language_model.backbone.layers.1.mixer.experts.0.gate_proj.weight",
        "language_model.backbone.layers.1.mixer.shared_experts.gate_proj.weight",
        "language_model.backbone.layers.1.mixer.experts.0.up_proj.bias",
        "language_model.backbone.layers.1.mixer.shared_experts.up_proj.bias",
        "language_model.backbone.layers.0.mixer.in_proj.weight_scale_2",
        "language_model.backbone.layers.0.mixer.in_proj.bias",
        "language_model.backbone.layers.0.mixer.unknown.weight",
        "language_model.backbone.layers.0.unknown.weight",
        "language_model.backbone.layers.0",
        "language_model.backbone.layers.0.",
        "language_model.backbone.embed_tokens.weight",
        "language_model.backbone.layers.7.self_attn.q_proj.weight",
        "language_model.backbone.layers.1.moe.gate.weight",
        "language_model.lm_head.weight_scale_2",
        "language_model.backbone.layers.00.norm.weight",
        "language_model.backbone.layers.+0.norm.weight",
        "language_model.backbone.layers.1.mixer.experts.00.up_proj.weight",
    ] {
        assert!(
            matches!(
                spec.key_plan(name),
                Err(inference_engine::EngError::Checkpoint(_))
            ),
            "{name}"
        );
    }
}

#[test]
fn key_plan_covers_pinned_real_index_when_enabled() {
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, BTreeSet};
    // Set NEM35_COMPLETE_INDEX to the pinned official-metadata/model.safetensors.index.json.
    let Some(path) = std::env::var_os("NEM35_COMPLETE_INDEX") else {
        return;
    };
    let bytes = std::fs::read(path).expect("pinned index file");
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "c368acdbcfe882236af1ec79df4a1fb6127e6ed9f742d59c01c21fe96ddc2741"
    );
    let index: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let map = index["weight_map"].as_object().expect("weight_map");
    let spec = NemotronTextSpec::from_config(&pinned_config()).unwrap();
    let mut classes = vec![BTreeSet::new(); 88];
    let mut counts = [0; 88];
    let mut experts = vec![BTreeMap::<u32, usize>::new(); 88];
    let mut globals = 0;
    let mut text = 0;
    for name in map.keys().filter(|name| {
        name.starts_with("language_model.backbone.") || name.starts_with("language_model.lm_head.")
    }) {
        let key = spec
            .key_plan(name)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        text += 1;
        if let Some(block) = key.block {
            counts[block] += 1;
            classes[block].insert(key.class);
            if let Some(expert) = key.expert {
                *experts[block].entry(expert).or_default() += 1;
            }
        } else {
            globals += 1;
        }
    }
    // The inventory conservatively marks the three lm_head tensors ambiguous.
    assert_eq!(text, 164_989);
    assert_eq!(globals, 5);
    for (block, kind) in spec.block_order().iter().enumerate() {
        assert_eq!(
            classes[block],
            kind.expected_key_classes().iter().copied().collect(),
            "block {block}"
        );
        let expected_count = if *kind == NemotronBlockKind::Moe {
            4109
        } else {
            13
        };
        assert_eq!(counts[block], expected_count, "block {block}");
        if *kind == NemotronBlockKind::Moe {
            assert_eq!(
                experts[block].keys().copied().collect::<Vec<_>>(),
                (0..512).collect::<Vec<_>>()
            );
            assert!(experts[block].values().all(|count| *count == 8));
        }
    }
}
