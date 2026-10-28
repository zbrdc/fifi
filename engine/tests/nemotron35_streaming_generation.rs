#![cfg(all(feature = "native-cuda", target_os = "linux"))]

use std::path::Path;

use inference_engine::family::nemotron35::runner;

const CHUNK: usize = 256;

#[test]
#[ignore = "requires the pinned checkpoint, CUDA, and an exclusive streamed-generation gate"]
fn full_prefill_chunk_releases_streamed_experts_through_the_generation_driver() {
    for (name, value) in [
        ("NEM35_CPU_EXPERTS", "1"),
        ("NEM35_GPU_MOE_LAYERS", "0"),
        ("NEM35_RESIDENT_DENSE", "1"),
        ("NEM35_CHUNK_EXPERTS", "gpu"),
        ("NEM35_CUDA_CHUNK_PREFILL", "256"),
    ] {
        assert_eq!(std::env::var(name).as_deref(), Ok(value), "{name}");
    }
    let model = std::env::var_os("NEM35_MODEL_DIR").expect("set NEM35_MODEL_DIR for this gate");
    let prompt = vec![1; CHUNK];
    // Keep the sealed 88-layer descriptor and the exact entry used by the timing binary.
    let generated = runner::generate_cuda(Path::new(&model), &prompt, 1, 64)
        .expect("one full streamed prefill chunk must release its selected experts");
    assert_eq!(generated.prompt, prompt);
    assert!(generated.produced.iter().all(|&id| id < 131_072));
    let hybrid = generated.hybrid.expect("hybrid streaming is enabled");
    assert_eq!(hybrid.cpu_layers.len(), 40);
    assert!(hybrid.gpu_layers.is_empty());
    println!(
        "streamed-generation gate: rows={CHUNK}, real_layers=88, produced={:?}",
        generated.produced
    );
}
