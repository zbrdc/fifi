//! Test support for the scale storage of the Q36 resident weights (lever Q36-G0).
//!
//! The default stores the scales and biases of the packed weights as F32. MLX casts BF16 scales
//! and biases to the F32 of the activation on every call of `qmm` and `gather_qmm`, and an F32
//! operand skips that cast. BF16 widens to F32 without loss, so the products do not change. The
//! knob selects the old BF16 storage for a comparison run. A Q36 result never satisfies Phase 0B
//! or Phase 1B.
//!
//! | Variable | Values | Default |
//! | --- | --- | --- |
//! | `Q36_SCALE_DTYPE` | `f32` widens the stored scales and biases at import, `bf16` keeps them as stored | `f32` |
//!
//! `bf16` needs the `native-fault-qualification` feature. Any other set value fails the test. The
//! knob applies to the real-checkpoint tests of `parity`, `route_parity` (both tests), and
//! `weights` (the load test). Each report carries the dtype, the census of the scale and bias
//! tensors, and the resident bytes after the load.

use sha2::{Digest, Sha256};

use super::decoder::ResidentDecoder;
use super::real::rss_kb;
use crate::{
    family::qwen36::host_load::HostWeights, kernels::MlxKernels, weights::StreamedCheckpoint,
};

/// The packed weights of the 35B model: the head, 30 x 5 DeltaNet, 10 x 4 attention, and 40 x 6
/// MoE weights (three shared and three expert stacks).
const PACKED_WEIGHTS: u64 = 431;

/// Reads `Q36_SCALE_DTYPE`: unset or `f32` is true, `bf16` is false, any other value panics. The
/// BF16 storage exists only with the `native-fault-qualification` feature.
pub(super) fn scales_f32_requested() -> bool {
    match std::env::var("Q36_SCALE_DTYPE") {
        Err(std::env::VarError::NotPresent) => true,
        Ok(value) if value == "f32" => true,
        #[cfg(feature = "native-fault-qualification")]
        Ok(value) if value == "bf16" => false,
        #[cfg(not(feature = "native-fault-qualification"))]
        Ok(value) if value == "bf16" => {
            panic!("Q36_SCALE_DTYPE=bf16 needs the native-fault-qualification feature")
        }
        other => panic!("Q36_SCALE_DTYPE must be unset, f32, or bf16, got {other:?}"),
    }
}

/// Builds the decoder that `Q36_SCALE_DTYPE` asks for and checks the dtype of every scale and
/// bias tensor: 862 tensors in 431 packed weights, all F32 (default) or all BF16. Returns the
/// decoder and a report object with the dtype, the census, and the memory after the load.
pub(super) fn new_decoder<'c>(
    kernels: MlxKernels,
    ckpt: &'c StreamedCheckpoint,
    host: HostWeights,
    context_limit: u32,
) -> (ResidentDecoder<'c>, serde_json::Value) {
    let f32_scales = scales_f32_requested();
    #[cfg(feature = "native-fault-qualification")]
    let decoder = if f32_scales {
        ResidentDecoder::new(kernels, ckpt, host, context_limit)
    } else {
        ResidentDecoder::new_bf16_scales(kernels, ckpt, host, context_limit)
    }
    .unwrap();
    #[cfg(not(feature = "native-fault-qualification"))]
    let decoder = ResidentDecoder::new(kernels, ckpt, host, context_limit).unwrap();
    let census = decoder.weights.scale_census();
    let (want_f32, want_bf16) = if f32_scales { (862, 0) } else { (0, 862) };
    assert_eq!(
        (
            census.weights,
            census.f32_parts,
            census.bf16_parts,
            census.other_parts
        ),
        (PACKED_WEIGHTS, want_f32, want_bf16, 0),
        "scale and bias dtypes do not follow Q36_SCALE_DTYPE"
    );
    let report = serde_json::json!({
        "dtype": if f32_scales { "f32" } else { "bf16" },
        "packed_weights": census.weights,
        "f32_parts": census.f32_parts, "bf16_parts": census.bf16_parts,
        "resident_bytes": decoder.weights.byte_len().unwrap(),
        "expert_page_bytes": decoder.weights.expert_page_bytes().unwrap(),
        "mlx_active_after_load": decoder.kernels.memory_stats().unwrap().active,
        "rss_kb_after_load": rss_kb(),
    });
    (decoder, report)
}

/// Adds the little-endian bytes of `values` to `hasher`.
pub(super) fn hash_f32s(hasher: &mut Sha256, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    hasher.update(&bytes);
}
