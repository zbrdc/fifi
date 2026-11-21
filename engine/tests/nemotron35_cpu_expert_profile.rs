//! Opt-in benchmark of the production mapped NVFP4 expert pool.

use inference_engine::family::nemotron35::runner::{CpuLayerTiming, time_cpu_experts};
use std::{env, path::PathBuf, process::Command};

#[test]
#[ignore = "requires a pinned Nemotron shard with every routed expert of the selected layer"]
fn profile_pinned_nemotron_mapped_experts() {
    let path = |name| PathBuf::from(env::var_os(name).expect(name));
    let integer = |name| env::var(name).expect(name).parse::<usize>().expect(name);
    let checkpoint = path("NEM35_CPU_PROFILE_CHECKPOINT");
    let threads = integer("NEM35_CPU_PROFILE_THREADS");
    let iterations = integer("NEM35_CPU_PROFILE_ITERATIONS");
    let layer = u32::try_from(integer("NEM35_CPU_PROFILE_LAYER")).unwrap();
    let counter = path("NEM35_CPU_PROFILE_PROC_IO");
    let io = || {
        let report = Command::new(&counter)
            .arg(std::process::id().to_string())
            .output()
            .unwrap();
        assert!(report.status.success());
        serde_json::from_slice::<serde_json::Value>(&report.stdout).unwrap()
    };

    assert!(threads > 0 && iterations >= 3);
    let before = io();
    let [timing]: [CpuLayerTiming; 1] =
        time_cpu_experts(&checkpoint, &[layer], threads, iterations)
            .unwrap()
            .try_into()
            .expect("exactly one layer");
    let after = io();
    assert_eq!(timing.warm_ms.len(), iterations);
    assert!(timing.warm_ms.iter().all(|&value| value > 0.0));
    assert!(timing.first.as_nanos() > 0);
    let delta = |key: &str| after[key].as_u64().unwrap() - before[key].as_u64().unwrap();
    println!(
        "nemotron_cpu_expert_profile {}",
        serde_json::json!({
            "layer": timing.layer,
            "threads": threads,
            "samples": timing.warm_ms.len(),
            "isa": format!("{:?}", timing.isa),
            "first_ms": timing.first.as_secs_f64() * 1e3,
            "p50_ms": timing.p50_ms(),
            "p90_ms": timing.p90_ms(),
            "disk_read_bytes": delta("bytes_read"),
            "pageins": delta("pageins"),
            "warm_ms": timing.warm_ms,
        })
    );
}
