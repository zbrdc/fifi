#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::sync::{Arc, Mutex};

use inference_engine::{
    EngError,
    kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::Dtype,
    weights::{ResidentWeights, WeightSource},
};

static GPU_TEST: Mutex<()> = Mutex::new(());

fn affine(bits: u32, group_size: u32) -> QuantizationSpec {
    QuantizationSpec {
        mode: QuantizationMode::Affine,
        bits,
        group_size,
    }
}

fn fixture_weight() -> PackedWeight {
    let source = ResidentWeights::from_bytes(Arc::from(
        include_bytes!("fixtures/packed.safetensors").as_slice(),
    ))
    .unwrap();
    PackedWeight::try_new(
        source.tensor("weight").unwrap(),
        source.tensor("scales").unwrap(),
        Some(source.tensor("biases").unwrap()),
        vec![2, 64],
        affine(4, 64),
    )
    .unwrap()
}

fn mxfp4_weight() -> PackedWeight {
    let words = [0x76543210u32, 0xfedcba98]
        .into_iter()
        .flat_map(|word| std::iter::repeat_n(word, 8))
        .flat_map(u32::to_le_bytes)
        .collect();
    let scales = vec![1u8, 2, 3, 4];
    PackedWeight::try_new(
        inference_engine::tensor::Tensor::try_from_host(Dtype::U32, vec![2, 8], words).unwrap(),
        inference_engine::tensor::Tensor::try_from_host(Dtype::U8, vec![2, 2], scales).unwrap(),
        None,
        vec![2, 64],
        QuantizationSpec {
            mode: QuantizationMode::Mxfp4,
            bits: 4,
            group_size: 32,
        },
    )
    .unwrap()
}

#[test]
fn import_packed_retains_the_fixture_components() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let weight = fixture_weight();
    let mut kernels = MlxKernels::try_new().unwrap();

    let resident = kernels.import_packed(&weight).unwrap();

    assert_eq!(resident.logical_shape(), weight.logical_shape());
    assert_eq!(resident.quantization(), weight.quantization());
    assert_eq!(resident.logical_elements(), 128);
    let payload = kernels.readback(resident.payload()).unwrap();
    assert_eq!(
        payload.host_bytes().unwrap(),
        weight.payload().host_bytes().unwrap()
    );
    let scales = kernels.readback(resident.scales()).unwrap();
    assert_eq!(
        scales.host_bytes().unwrap(),
        weight.scales().host_bytes().unwrap()
    );
    let biases = kernels.readback(resident.biases().unwrap()).unwrap();
    assert_eq!(
        biases.host_bytes().unwrap(),
        weight.biases().unwrap().host_bytes().unwrap()
    );
}

#[test]
fn import_packed_imports_mxfp_scales_without_biases() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let weight = mxfp4_weight();
    let mut kernels = MlxKernels::try_new().unwrap();

    let resident = kernels.import_packed(&weight).unwrap();

    assert!(resident.biases().is_none());
    assert_eq!(resident.logical_elements(), 128);
    let scales = kernels.readback(resident.scales()).unwrap();
    assert_eq!(
        scales.host_bytes().unwrap(),
        weight.scales().host_bytes().unwrap()
    );
}

#[test]
fn resident_tensors_stay_with_their_adapter_owner() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let weight = fixture_weight();
    let mut kernels = MlxKernels::try_new().unwrap();
    let resident = kernels.import_packed(&weight).unwrap();
    let mut other = MlxKernels::try_new().unwrap();

    let result = other.readback(resident.payload());

    assert!(matches!(
        result,
        Err(EngError::BadRequest(message)) if message.contains("another native owner")
    ));
}
