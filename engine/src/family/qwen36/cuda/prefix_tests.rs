//! CPU tests of the pure prefix helpers: the pinned host-byte charge of an entry in both KV
//! modes and its refusal on an absurd count. The device round trips are `prefix_device_tests.rs`.
use super::super::host::HostShape;
use super::dims::{KvMode, StepDims, StepOptions};
use super::prefix::entry_bytes;
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

/// The pinned charge (packet IMPL-CUDA-SERVE-2 G1/G2): 65,863,680 B of DeltaNet export plus
/// 20,480 B per position at Fp16 and 10,880 B per position at Int8G32.
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
    }
}

#[test]
fn an_absurd_positions_count_is_a_refusal_not_a_panic() {
    let d = dims();
    let layout = StateLayout::new(&d, &options(KvMode::Fp16, 1_024)).expect("layout");
    assert!(entry_bytes(&layout, usize::MAX).is_err());
}
