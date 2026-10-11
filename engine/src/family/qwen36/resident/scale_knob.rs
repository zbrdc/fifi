//! Test support for the activation dtype of the Q36 resident decoder (packet Q36-BF16-a).
//!
//! The product lever is the `activation_dtype` setting of the engine entry (`settings.rs`): `bf16`
//! keeps the stored BF16 scales and biases and runs BF16 activations with converted norms, `f32`
//! keeps the G0 widening, and the decoder constructor refuses any other pairing. The ignored
//! real-checkpoint runs have no engine entry, so this file reads the same lever from one knob and
//! builds the decoder through the product constructor. A Q36 result never satisfies Phase 0B or
//! Phase 1B.
//!
//! | Variable | Values | Default |
//! | --- | --- | --- |
//! | `Q36_ACTIVATION` | `f32` is the qualification control, `bf16` builds the BF16 decoder | `f32` |
//!
//! Any other set value fails the test. The knob applies to the real-checkpoint tests of `parity`,
//! `route_parity` (both tests), and `prefill/parity` (its two parity gates and the prompt-speed
//! test; packet Q36-BF16-b). The weights load test keeps `Q36_SCALE_DTYPE`, the older knob
//! that selects the scale import alone, because it measures both import forms at the weights level;
//! as a decoder pairing that combination is refused.

use sha2::{Digest, Sha256};

use super::decoder::ResidentDecoder;
use super::real::{activation_from_env, rss_kb};
use crate::{
    family::qwen36::host_load::HostWeights,
    family::qwen36::settings::{Activation, activation_name},
    kernels::MlxKernels,
    weights::StreamedCheckpoint,
};

/// The packed weights of the 35B model: the head, 30 x 5 DeltaNet, 10 x 4 attention, and 40 x 6
/// MoE weights (three shared and three expert stacks).
const PACKED_WEIGHTS: u64 = 431;

/// Reads `Q36_SCALE_DTYPE`: unset or `f32` is true, `bf16` is false, any other value panics. The
/// knob selects the scale import of the weights load test only (see the module doc).
pub(super) fn scales_f32_requested() -> bool {
    match std::env::var("Q36_SCALE_DTYPE") {
        Err(std::env::VarError::NotPresent) => true,
        Ok(value) if value == "f32" => true,
        Ok(value) if value == "bf16" => false,
        other => panic!("Q36_SCALE_DTYPE must be unset, f32, or bf16, got {other:?}"),
    }
}

/// Builds the decoder that `Q36_ACTIVATION` asks for, through the product constructor, and checks
/// the dtype of every scale and bias tensor: 862 tensors in 431 packed weights, all F32 (control)
/// or all BF16. Returns the decoder and the activation it holds, for the report.
pub(super) fn new_decoder<'c>(
    kernels: MlxKernels,
    ckpt: &'c StreamedCheckpoint,
    host: HostWeights,
    context_limit: u32,
) -> (ResidentDecoder<'c>, Activation) {
    let activation = activation_from_env();
    let decoder =
        ResidentDecoder::new_with_activation(kernels, ckpt, host, context_limit, activation)
            .unwrap();
    let census = decoder.weights.scale_census();
    let (want_f32, want_bf16) = match activation {
        Activation::F32 => (862, 0),
        Activation::Bf16 => (0, 862),
    };
    assert_eq!(
        (
            census.weights,
            census.f32_parts,
            census.bf16_parts,
            census.other_parts
        ),
        (PACKED_WEIGHTS, want_f32, want_bf16, 0),
        "scale and bias dtypes do not follow Q36_ACTIVATION"
    );
    assert_eq!(
        decoder.weights.consts.act,
        match activation {
            Activation::F32 => crate::tensor::Dtype::F32,
            Activation::Bf16 => crate::tensor::Dtype::Bf16,
        },
        "the decoder holds the activation dtype"
    );
    let report = serde_json::json!({
        "activation_dtype": activation_name(activation),
        "packed_weights": census.weights,
        "f32_parts": census.f32_parts, "bf16_parts": census.bf16_parts,
        "resident_bytes": decoder.weights.byte_len().unwrap(),
        "expert_page_bytes": decoder.weights.expert_page_bytes().unwrap(),
        "mlx_active_after_load": decoder.kernels.memory_stats().unwrap().active,
        "rss_kb_after_load": rss_kb(),
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    (decoder, activation)
}

/// Adds the little-endian bytes of `values` to `hasher`.
pub(super) fn hash_f32s(hasher: &mut Sha256, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    hasher.update(&bytes);
}
