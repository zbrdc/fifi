//! CPU tests of the pure prefix helpers: the pinned host-byte charge of an entry in both KV
//! modes and its refusal on an absurd count. The device round trips are `prefix_device_tests.rs`.
use super::super::host::HostShape;
use super::dims::{KvMode, StepDims, StepOptions};
use super::prefix::{CheckpointPayload, CudaSavedPrefix, entry_bytes};
use super::state::{
    StateIdentity,
    export::{DeltaExport, ExportedState, KvExport, plane_sizes},
};
use super::state_layout::StateLayout;
use crate::family::project::cuda::WeightLayout;
use crate::family::{QWEN3_6_35B_A3B, gguf_model::Inventory};

fn dims() -> StepDims {
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).expect("shape");
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    StepDims::new(&shape, &inventory).expect("dims")
}

fn options(mode: KvMode, ctx: usize) -> StepOptions {
    StepOptions {
        kv_mode: mode,
        context_tokens: ctx,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Tiled,
    }
}

/// The pinned charge: 65,863,680 B of DeltaNet export plus 20,480 B per position at Fp16 and
/// 10,880 B per position at Int8G32.
#[test]
fn entry_bytes_matches_the_pinned_host_charge_of_both_modes() {
    let d = dims();
    for mode in [KvMode::Fp16, KvMode::Int8G32] {
        let layout = StateLayout::new(&d, &options(mode, 40_960)).expect("layout");
        let per_token = match mode {
            KvMode::Fp16 => 20_480_usize,
            KvMode::Int8G32 => 10_880_usize,
        };
        for t in [0_usize, 1, 37, 32_767] {
            assert_eq!(
                entry_bytes(&layout, t).expect("entry bytes"),
                65_863_680 + per_token * t,
                "{mode:?} at {t} positions"
            );
        }
    }
}

#[test]
fn entry_bytes_equals_the_summed_lengths_of_a_zero_export() {
    let d = dims();
    for mode in [KvMode::Fp16, KvMode::Int8G32] {
        let layout = StateLayout::new(&d, &options(mode, 1_024)).expect("layout");
        let t = 37;
        let exported = ExportedState {
            identity: StateIdentity::new(&d, mode),
            tokens: t,
            delta: (0..layout.delta.layers)
                .map(|_| DeltaExport {
                    recurrent: vec![0.0; layout.delta.rec_values],
                    conv: vec![0.0; layout.delta.conv_values],
                })
                .collect(),
            kv: (0..layout.kv.layers)
                .map(|o| KvExport {
                    planes: plane_sizes(&layout.kv, o, t)
                        .expect("sizes")
                        .into_iter()
                        .map(|n| vec![0u8; n])
                        .collect(),
                })
                .collect(),
        };
        let delta: usize = exported
            .delta
            .iter()
            .map(|layer| (layer.recurrent.len() + layer.conv.len()) * size_of::<f32>())
            .sum();
        let kv: usize = exported
            .kv
            .iter()
            .map(|layer| layer.planes.iter().map(Vec::len).sum::<usize>())
            .sum();
        assert_eq!(
            entry_bytes(&layout, t).expect("entry bytes"),
            delta + kv,
            "{mode:?}: the charge equals the export's own bytes"
        );
        assert_eq!(
            CudaSavedPrefix::Pageable(exported).payload_bytes(),
            (delta + kv) as u64
        );
    }
}

#[test]
fn an_absurd_positions_count_is_a_refusal_not_a_panic() {
    let d = dims();
    let layout = StateLayout::new(&d, &options(KvMode::Fp16, 1_024)).expect("layout");
    assert!(entry_bytes(&layout, usize::MAX).is_err());
}

#[test]
fn pageable_peak_sums_shards_and_reserves_only_one_copy_temporary() {
    use super::prefix::{HOST_HEADROOM, pageable_required};
    for mode in [KvMode::Fp16, KvMode::Int8G32] {
        let layout = StateLayout::new(&dims(), &options(mode, 40_960)).unwrap();
        for n in [0, 1, 32_767] {
            let one = pageable_required([(&layout, n)].into_iter()).unwrap();
            let two = pageable_required([(&layout, n), (&layout, n)].into_iter()).unwrap();
            let payload = entry_bytes(&layout, n).unwrap();
            assert_eq!(two - one, payload);
            let conversion =
                layout.delta.rec_values.max(layout.delta.conv_values) * size_of::<f32>();
            let readback = layout
                .kv
                .prefix_ranges(0, n)
                .unwrap()
                .iter()
                .map(|run| run.bytes.min(crate::kernels::CHUNK_LIMIT))
                .max()
                .unwrap();
            assert_eq!(one, HOST_HEADROOM + payload + conversion.max(readback));
        }
        let narrow = StateLayout::new_with_kv_heads(&dims(), &options(mode, 40_960), 1).unwrap();
        let n = 32_767;
        for (first, second) in [(&narrow, &layout), (&layout, &narrow)] {
            let first_peak = pageable_required([(first, n)].into_iter()).unwrap();
            let second_peak = pageable_required([(second, n)].into_iter()).unwrap();
            let peak = pageable_required([(first, n), (second, n)].into_iter()).unwrap();
            assert_eq!(
                peak,
                first_peak.max(entry_bytes(first, n).unwrap() + second_peak)
            );
        }
        assert!(pageable_required([(&layout, usize::MAX)].into_iter()).is_err());
    }
}

#[test]
#[ignore = "requires a Linux process with MemoryMax=2G and more than 1.8 GiB initially available"]
fn process_memory_limit_2_gib_admits_one_large_pageable_shard_and_refuses_two() {
    let budget = crate::family::host_admit::HostAdmission::probe("test", "test").unwrap();
    assert!(((1800 << 20)..=(2 << 30)).contains(&budget.available_budget()));
    let dims = dims();
    let positions = 49_152;
    let options = options(KvMode::Fp16, positions);
    let heads = dims.shape.attn.heads_kv;
    let left = StateLayout::new_with_kv_heads(&dims, &options, heads.div_ceil(2)).unwrap();
    let right = StateLayout::new_with_kv_heads(&dims, &options, heads / 2).unwrap();
    let shards = [(&left, positions), (&right, positions)];
    for shard in shards {
        assert!(super::prefix::admit_pageable(std::iter::once(shard)).unwrap());
    }
    assert!(!super::prefix::admit_pageable(shards.into_iter()).unwrap());
}

#[test]
fn the_save_decision_admits_at_the_boundary_and_skips_with_a_line() {
    use super::prefix::admit_or_log;
    // Ample room admits.
    assert!(admit_or_log(10, usize::MAX));
    // A shortfall skips; the boundary admits.
    assert!(!admit_or_log(10, 9));
    assert!(admit_or_log(10, 10));
}
