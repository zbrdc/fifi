use std::{sync::Arc, time::Instant};

use inference_engine::{
    kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::{Dtype, Tensor},
    weights::{ResidentWeights, WeightSource},
};

use super::{checked, sample};

fn fixture() -> (Tensor, PackedWeight) {
    let source = ResidentWeights::from_bytes(Arc::from(
        include_bytes!("../../../engine/tests/fixtures/packed.safetensors").as_slice(),
    ))
    .expect("fixture storage");
    let weight = PackedWeight::try_new(
        source.tensor("weight").expect("weight"),
        source.tensor("scales").expect("scales"),
        Some(source.tensor("biases").expect("biases")),
        vec![2, 64],
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 4,
            group_size: 64,
        },
    )
    .expect("packed fixture");
    let mut values = vec![1.0f32; 64];
    values.push(1.0);
    values.extend([0.0; 63]);
    let activation = Tensor::try_from_host(
        Dtype::F32,
        vec![2, 64],
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
    .expect("activation fixture");
    (activation, weight)
}

fn f32_stage_bytes(tensor: &Tensor) -> u64 {
    tensor
        .shape()
        .iter()
        .map(|&dimension| u64::from(dimension))
        .product::<u64>()
        * 4
}

fn cycle(activation: &Tensor, weight: &PackedWeight) -> u128 {
    let mut adapter = MlxKernels::try_new().expect("native adapter");
    let start = Instant::now();
    let output = adapter
        .dequant_matmul(activation, weight)
        .expect("native matmul");
    let elapsed_us = start.elapsed().as_micros();
    let expected = [-288.0f32, -1216.0, -8.0, -12.0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    if output.shape() != [2, 2] || output.host_bytes() != Some(expected.as_slice()) {
        eprintln!("native adapter output differs from the pinned fixture");
        std::process::exit(2);
    }
    drop(output);
    drop(adapter);
    checked(unsafe { mlx_sys::mlx_clear_cache() }, "clear native cache");
    elapsed_us
}

pub(super) fn run() {
    let (activation, weight) = fixture();
    let stage_bytes = activation.byte_len()
        + weight.payload().byte_len()
        + f32_stage_bytes(weight.scales())
        + f32_stage_bytes(weight.biases().expect("biases"));
    println!(
        "{{\"stage\":\"adapter_identity\",\"typed_stage_bytes\":{stage_bytes},\"output_bytes\":16,\"cycles\":128}}"
    );
    sample("adapter_start", 0);
    checked(unsafe { mlx_sys::mlx_reset_peak_memory() }, "reset peak");
    for index in 1..=128 {
        let elapsed_us = cycle(&activation, &weight);
        if matches!(index, 1 | 32 | 64 | 128) {
            sample(&format!("adapter_cycle_{index}"), elapsed_us);
        }
    }
    drop(weight);
    drop(activation);
    sample("adapter_released", 0);
}
