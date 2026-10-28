use super::super::super::profile;
use super::*;
use crate::family::flint::cuda_ffn::real_model;

#[test]
#[ignore = "requires fresh admitted GPU, bounded cgroup v2 and pinned GGUF; N0 phase diagnostic only"]
fn selected_cold_decode_phase_profile() {
    let (file, header, identity) = real_model::model();
    let prefixes = real_model::prefixes();
    let ctx = CudaContext::new().expect("required CUDA device");
    let stream = ctx.stream().unwrap();
    let mut decode = CudaDecode::new_selected(&ctx, &stream, (&file, &header), 8, 0).unwrap();
    assert_eq!(decode.fetch_stats().resident_experts, 0);
    let mut next = None;
    for (position, &token) in prefixes[0].input_ids.iter().enumerate() {
        // Raw stderr orders readiness before STEP; captured stdout cannot prove this order.
        if position == 0 {
            selected::first_step_marker(&mut std::io::stderr().lock()).unwrap();
        }
        let before = profile_counters::sample();
        let capture = profile::Capture::start();
        {
            let _step = profile::wall("decode_step");
            next = Some(decode.step(token).unwrap());
        }
        let report = capture.finish();
        let host_delta = profile_counters::delta(before, profile_counters::sample());
        let stats = decode.fetch_stats();
        let prepared_enabled = decode.prepared_enabled();
        // These are resolved view bytes by source policy, not OS-cache misses or DMA bytes.
        let prepared_hit_bytes = if prepared_enabled {
            stats.payload_bytes
        } else {
            0
        };
        let prepared_miss_bytes = if prepared_enabled {
            0
        } else {
            stats.payload_bytes
        };
        assert_eq!(
            prepared_hit_bytes + prepared_miss_bytes,
            stats.payload_bytes
        );
        assert!(!prepared_enabled || prepared_miss_bytes == 0);
        assert_eq!(stats.cold_experts.len(), 47 * 6);
        assert_eq!(stats.payload_bytes, 47 * 6 * 1_769_472);
        assert_eq!(decode.tokens(), position + 1);
        assert!(report.gemm_calls.iter().sum::<u64>() > 0);
        for name in [
            "decode_step",
            "fetch",
            "decode_head",
            "decode_host_qk_write",
        ] {
            assert!(report.wall_ns[name] > 0, "missing phase {name}");
        }
        println!(
            "SELECTED_PHASE {}",
            serde_json::json!({
                "position":position, "next_id":next, "report":report,
                "cold_experts":stats.cold_experts.len(), "payload_bytes":stats.payload_bytes,
                "prepared_enabled":prepared_enabled,
                "prepared_hit_bytes":prepared_hit_bytes, "prepared_miss_bytes":prepared_miss_bytes,
                "host_delta":host_delta,
                "host_counter_order":["process_read_bytes", "cgroup_pgscan", "cgroup_pgsteal", "cgroup_workingset_refault_file", "cgroup_max_events"],
                "gemm_phase_order":["other_attention_head_selected_shared_cold", "dense", "tagged_shared", "tagged_routed"],
                "wall_scope":"inclusive; head includes load and head GEMM; do not sum nested wall/GPU spans",
                "gpu_scope":"instrumented Frame projections, not whole-device compute",
                "unclassified_remainder":"status waits, buffer-drop sync and transfers outside named spans; not attributed by subtraction",
                "constructor_separate":true, "head_rows":1, "logits_written":false,
                "shared_correctness_only":true, "qualification":false
            })
        );
    }
    assert_eq!(next.unwrap() as usize, real_model::FIRST_TOKENS[0]);
    drop(decode);
    gguf::verify_prepared_identity(&file, &identity).unwrap();
}
