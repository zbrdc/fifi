use super::{Attention, TextSpec};
use serde_json::{Value, json};

fn release_config() -> Value {
    json!({
        "model_type": "inkling2_loop",
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "eos_token_id": 200006,
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        "text_config": {
            "model_type": "inkling2_loop_text", "model_max_length": 1048576,
            "hidden_size": 2304, "num_hidden_layers": 24, "vocab_size": 201024,
            "unpadded_vocab_size": 200058,
            "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128,
            "swa_num_attention_heads": 24, "swa_num_key_value_heads": 8, "swa_head_dim": 128,
            "sliding_window_size": 1024,
            "local_layer_ids": [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
            "n_layers_per_attn_res_block": 6,
            "n_routed_experts": 128, "num_experts_per_tok": 8, "n_shared_experts": 0,
            "intermediate_size": 1152,
            "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3,
                            "gate_rank": 64, "is_varloop": false}
        }
    })
}

fn spec(config: &Value) -> Result<TextSpec, crate::EngError> {
    TextSpec::from_config(&serde_json::to_vec(config).unwrap())
}

#[test]
fn reuses_core_weights_and_allocates_each_visit_its_own_state() {
    let spec = spec(&release_config()).unwrap();
    let visits = spec.visits();
    assert_eq!(spec.stored_layers(), 24);
    assert_eq!(visits.len(), 48);
    let weights = visits
        .iter()
        .map(|visit| visit.weight_layer)
        .collect::<Vec<_>>();
    assert_eq!(
        weights,
        (0..6)
            .chain((0..3).flat_map(|_| 6..18))
            .chain(18..24)
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        visits
            .iter()
            .map(|visit| visit.state_slot)
            .collect::<Vec<_>>(),
        (0..48).collect::<Vec<_>>(),
    );
    assert_eq!(
        visits
            .iter()
            .filter(|visit| visit.attention == Attention::Global)
            .count(),
        8
    );
}

#[test]
fn caps_only_local_bf16_cache_at_the_attention_window() {
    let spec = spec(&release_config()).unwrap();
    let plan = spec.kv_plan(8192).unwrap();
    assert_eq!((plan.local_visits, plan.global_visits), (40, 8));
    assert_eq!(plan.windowed_bytes, 416 * 1024 * 1024);
    assert_eq!(plan.all_full_bytes, 1536 * 1024 * 1024);
    let short = spec.kv_plan(512).unwrap();
    assert_eq!(short.windowed_bytes, short.all_full_bytes);
    assert!(spec.kv_plan(0).is_err());
    assert!(spec.kv_plan(1048577).is_err());
}

#[test]
fn rejects_variable_loops_invalid_head_geometry_and_duplicate_local_layers() {
    let mut config = release_config();
    config["text_config"]["loop_config"]["is_varloop"] = json!(true);
    assert!(spec(&config).is_err());
    let mut config = release_config();
    config["text_config"]["swa_num_key_value_heads"] = json!(7);
    assert!(spec(&config).is_err());
    let mut config = release_config();
    config["text_config"]["local_layer_ids"] = json!([0, 0]);
    assert!(spec(&config).is_err());
}

#[test]
fn text_backends_require_a_single_supported_opt_in() {
    use super::TextBackend::{self, Cpu, Cuda};

    assert!(matches!(TextBackend::select(false, false), Ok(None)));
    assert!(TextBackend::select(true, true).is_err());
    if cfg!(target_pointer_width = "64") {
        assert!(matches!(TextBackend::select(true, false), Ok(Some(Cpu))));
    } else {
        assert!(TextBackend::select(true, false).is_err());
    }
    if cfg!(all(
        feature = "native-cuda",
        target_os = "linux",
        target_pointer_width = "64"
    )) {
        assert!(matches!(TextBackend::select(false, true), Ok(Some(Cuda))));
    } else {
        assert!(TextBackend::select(false, true).is_err());
    }
}
