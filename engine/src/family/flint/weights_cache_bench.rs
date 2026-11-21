//! Measure staging and the resident grouped projection on the same Flint weights.

use super::FlintCheckpoint;
use crate::family::flint::{
    mlx_forward::{disk_experts, grouped_experts},
    mlx_model::SwiGlu,
};
use crate::kernels::{MlxKernels, PackedWeight};
use crate::tensor::{Dtype, ExecutionTensor, Tensor};

fn ids(ids: &[u32]) -> Tensor {
    Tensor::try_from_host(
        Dtype::U32,
        vec![ids.len() as u32],
        ids.iter().flat_map(|id| id.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn stage(
    kernels: &mut MlxKernels,
    packed: &[PackedWeight; 3],
    ids: &Tensor,
) -> (SwiGlu, ExecutionTensor) {
    let (weights, indices) = kernels
        .import_packed_many_with(&[&packed[0], &packed[1], &packed[2]], &[ids])
        .unwrap();
    let [gate, up, down] = weights.try_into().ok().unwrap();
    let [local] = indices.try_into().ok().unwrap();
    (SwiGlu { gate, up, down }, local)
}

#[test]
#[ignore = "requires the published local Flint shards in FIFI_FLINT_CACHE_MODEL on Apple Silicon"]
fn flint_mlx_expert_cache_tiers() {
    let path = std::path::PathBuf::from(std::env::var_os("FIFI_FLINT_CACHE_MODEL").unwrap());
    let checkpoint = FlintCheckpoint::open(&path).unwrap();
    let _serial = crate::native_test_lock();
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.allow_bf16(true);
    kernels.set_cache_limit(64 << 20).unwrap();

    let stem = "model.layers.24.mlp.switch_mlp";
    for rows in [1_u32, 4, 16] {
        let input = Tensor::try_from_host(
            Dtype::Bf16,
            vec![rows, 2048],
            (0..rows * 2048)
                .flat_map(|index| {
                    let f = ((index % 37) as f32 - 18.0) / 32.0;
                    ((f.to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect(),
        )
        .unwrap();
        let input = kernels.import_tensor(&input).unwrap();
        let chosen: Vec<u32> = (0..rows * 6).map(|i| (i * 73 + 24) % 256).collect();
        let mut unique = chosen.clone();
        unique.sort_unstable();
        let local: Vec<u32> = chosen
            .iter()
            .map(|id| unique.binary_search(id).unwrap() as u32)
            .collect();
        let local = ids(&local);
        let packed = ["gate_proj", "up_proj", "down_proj"].map(|projection| {
            checkpoint
                .packed_experts(&format!("{stem}.{projection}"), &unique)
                .unwrap()
        });
        let (resident, resident_ids) = stage(&mut kernels, &packed, &local);
        let resident_ids = kernels.bound_indices(&resident_ids, rows * 6).unwrap();
        let disk_ids = kernels.import_tensor(&ids(&chosen)).unwrap();
        let disk_ids = kernels.bound_indices(&disk_ids, 256).unwrap();
        let run = |k: &mut MlxKernels, mode| match mode {
            "resident" => {
                let output = grouped_experts(k, &resident, &input, &resident_ids).unwrap();
                k.evaluate(&[&output]).unwrap();
                output
            }
            "host_packed" => {
                let (weights, indices) = stage(k, &packed, &local);
                let indices = k.bound_indices(&indices, rows * 6).unwrap();
                let output = grouped_experts(k, &weights, &input, &indices).unwrap();
                k.evaluate(&[&output]).unwrap();
                output
            }
            "file_warm" => disk_experts(k, &checkpoint, stem, &input, &disk_ids, false).unwrap(),
            _ => unreachable!(),
        };
        let expected = run(&mut kernels, "resident");
        let expected = kernels.readback(&expected).unwrap();
        assert!(expected.host_bytes().unwrap().iter().any(|&byte| byte != 0));
        let modes = ["resident", "host_packed", "file_warm"];
        for mode in modes {
            let _ = run(&mut kernels, mode);
        }
        for repetition in 0..9 {
            for offset in 0..modes.len() {
                let mode = modes[(repetition + offset) % modes.len()];
                let name = format!("flint_cache_{rows}_{mode}");
                let output = kernels.profile_fixture(&name, |k| run(k, mode));
                let result = kernels.readback(&output).unwrap();
                assert_eq!(result.host_bytes(), expected.host_bytes(), "{name}");
            }
        }
    }
}
