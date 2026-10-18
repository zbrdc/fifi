use std::sync::Arc;

use inference_engine::{
    kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::{Dtype, Tensor},
    weights::{ResidentWeights, WeightSource},
};

unsafe extern "C" {
    fn inference_gpu_stream_count(count: *mut usize) -> i32;
}

fn gpu_streams() -> usize {
    let mut count = 0;
    assert_eq!(unsafe { inference_gpu_stream_count(&mut count) }, 0);
    count
}

fn completed_cycle() {
    let result = {
        let source = ResidentWeights::from_bytes(Arc::from(
            include_bytes!("../../../engine/tests/fixtures/packed.safetensors").as_slice(),
        ))
        .unwrap();
        let weight = PackedWeight::try_new(
            source.tensor("weight").unwrap(),
            source.tensor("scales").unwrap(),
            Some(source.tensor("biases").unwrap()),
            vec![2, 64],
            QuantizationSpec {
                mode: QuantizationMode::Affine,
                bits: 4,
                group_size: 64,
            },
        )
        .unwrap();
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
        .unwrap();
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.dequant_matmul(&activation, &weight).unwrap()
    };
    assert_eq!(result.dtype(), Dtype::F32);
    assert_eq!(result.shape(), [2, 2]);
    let expected = [-288.0f32, -1216.0, -8.0, -12.0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(result.host_bytes().unwrap(), expected);
}

fn main() {
    completed_cycle();
    let baseline = gpu_streams();
    println!("{{\"stage\":\"warmup\",\"gpu_streams\":{baseline}}}");
    for cycle in 1..=128 {
        completed_cycle();
        let observed = gpu_streams();
        println!("{{\"stage\":\"same_thread\",\"cycle\":{cycle},\"gpu_streams\":{observed}}}");
        assert_eq!(observed, baseline, "GPU stream registrations grew");
    }
    for cycle in 1..=32 {
        std::thread::spawn(completed_cycle).join().unwrap();
        let observed = gpu_streams();
        println!(
            "{{\"stage\":\"sequential_thread\",\"cycle\":{cycle},\"gpu_streams\":{observed}}}"
        );
        assert_eq!(observed, baseline, "GPU stream registrations grew");
    }
}
