//! Flint resident weight planning and borrowed native views; upload remains staged.
#![allow(
    dead_code,
    reason = "resident descriptors await the staged CUDA loader"
)]

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
#[path = "exl3_resident/native/cache.rs"]
mod cache;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) use cache::HostBudget;
mod plan;
#[cfg(test)]
use plan::plan_resident;

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod native;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
#[allow(
    unused_imports,
    reason = "real uploaded fixture for staged FFN integration tests"
)]
pub(crate) use native::load_flint_fixture;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[allow(
    unused_imports,
    reason = "native API staged for Flint dispatch and loader"
)]
pub(crate) use native::{
    FlintExl3Resident, FlintExpertBank, FlintExpertView, FlintFfnView, FlintLayerView,
    load_flint_resident,
};

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use crate::family::project::cuda::Reserve;
use crate::weights::exl3::flint_pack::Plan;

#[derive(Clone, Copy)]
pub(crate) struct FlintResidentLimits {
    pub(crate) max_staging_bytes: u64,
    pub(crate) max_weight_bytes: u64,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(crate) reserve: Reserve,
}

struct ResidentPlan {
    layout: Plan,
    layers: Vec<LayerRoles>,
    embedding: usize,
    final_norm: usize,
    head: usize,
    stored_payload_bytes: u64,
    // Requested F32/packed payload; allocation padding is admitted by the upload slice.
    converted_payload_bytes: u64,
}

struct LayerRoles {
    qkvo: [usize; 4],
    norms: [usize; 4],
    sinks: usize,
    ffn: FfnRoles,
}

enum FfnRoles {
    Dense([usize; 3]),
    Moe {
        router: usize,
        experts: Vec<[usize; 3]>,
        shared: [usize; 3],
        routing_weight_multiplier: f32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EngError,
        weights::exl3::{
            Checkpoint,
            flint_pack::{Geometry, PackBuilder},
        },
    };

    #[test]
    fn resident_plan_enforces_payload_and_audit_staging_limits() {
        let directory = PackBuilder::tiny().write();
        let checkpoint = Checkpoint::open(directory.path()).unwrap();
        let geometry = Geometry::from_config(&checkpoint.model_config).unwrap();
        for (max_staging_bytes, max_weight_bytes) in [(1, 16 << 20), (16 << 20, 0)] {
            let result = plan_resident(
                directory.path(),
                &checkpoint,
                &geometry,
                FlintResidentLimits {
                    max_staging_bytes,
                    max_weight_bytes,
                    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                    reserve: Reserve::default(),
                },
            );
            assert!(
                matches!(&result, Err(EngError::MemoryPlan(_))),
                "staging={max_staging_bytes}, weight={max_weight_bytes}: {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn resident_plan_requires_the_shared_complete_pack_audit() {
        let mut pack = PackBuilder::tiny();
        assert!(
            pack.tensors
                .remove("model.layers.2.mlp.experts.3.up_proj.mul1")
                .is_some()
        );
        let directory = pack.write();
        let checkpoint = Checkpoint::open(directory.path()).unwrap();
        let geometry = Geometry::from_config(&checkpoint.model_config).unwrap();
        let result = plan_resident(
            directory.path(),
            &checkpoint,
            &geometry,
            FlintResidentLimits {
                max_staging_bytes: 16 << 20,
                max_weight_bytes: 16 << 20,
                #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                reserve: Reserve::default(),
            },
        );
        assert!(
            matches!(result, Err(EngError::Checkpoint(message)) if message.contains("schema tensor(s) missing"))
        );
    }

    #[test]
    fn resident_plan_binds_every_role_and_charges_converted_payload() {
        let pack = PackBuilder::tiny();
        let stored = pack
            .tensors
            .values()
            .map(|e| e.bytes.len() as u64)
            .sum::<u64>();
        let converted = pack
            .tensors
            .iter()
            .filter(|(name, _)| !name.ends_with(".mul1"))
            .map(|(name, entry)| {
                let bytes = entry.bytes.len() as u64;
                if name.ends_with(".trellis") {
                    bytes
                } else {
                    bytes * 2
                }
            })
            .sum::<u64>();
        let directory = pack.write();
        let checkpoint = Checkpoint::open(directory.path()).unwrap();
        let geometry = Geometry::from_config(&checkpoint.model_config).unwrap();
        let plan = plan_resident(
            directory.path(),
            &checkpoint,
            &geometry,
            FlintResidentLimits {
                max_staging_bytes: 16 << 20,
                max_weight_bytes: 16 << 20,
                #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                reserve: Reserve::default(),
            },
        )
        .expect("admit the complete tiny Flint pack through the shared reader");
        assert_eq!(plan.layers.len(), 3);
        assert_eq!(plan.layout.modules.len(), 46);
        assert_eq!(plan.stored_payload_bytes, stored);
        assert_eq!(plan.converted_payload_bytes, converted);
        assert_eq!(plan.layout.modules[plan.head].prefix, "lm_head");
        assert_eq!(plan.layout.modules[plan.head].words, 80);
        assert_eq!(
            plan.layout.plain[plan.embedding].name,
            "model.embed_tokens.weight"
        );
        assert_eq!(plan.layout.plain[plan.final_norm].name, "model.norm.weight");
        for (index, layer) in plan.layers.iter().enumerate() {
            for (role, matrix) in ["q", "k", "v", "o"].into_iter().zip(layer.qkvo) {
                assert_eq!(
                    plan.layout.modules[matrix].prefix,
                    format!("model.layers.{index}.self_attn.{role}_proj")
                );
            }
            assert_eq!(plan.layout.modules[layer.qkvo[0]].n, 256);
            for (name, norm) in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ]
            .into_iter()
            .zip(layer.norms)
            {
                assert_eq!(
                    plan.layout.plain[norm].name,
                    format!("model.layers.{index}.{name}")
                );
            }
            assert_eq!(
                plan.layout.plain[layer.sinks].name,
                format!("model.layers.{index}.self_attn.softmax_offset")
            );
            assert_eq!(plan.layout.plain[layer.sinks].shape, [2]);
        }
        let FfnRoles::Dense(dense) = &plan.layers[0].ffn else {
            panic!("expected dense Flint layer")
        };
        for (role, matrix) in ["gate", "up", "down"].into_iter().zip(*dense) {
            assert_eq!(
                plan.layout.modules[matrix].prefix,
                format!("model.layers.0.mlp.{role}_proj")
            );
        }
        for (index, layer) in plan.layers.iter().enumerate().skip(1) {
            let FfnRoles::Moe {
                router,
                experts,
                shared,
                routing_weight_multiplier,
            } = &layer.ffn
            else {
                panic!("expected sparse Flint layer")
            };
            assert_eq!(
                plan.layout.plain[*router].name,
                format!("model.layers.{index}.mlp.gate.weight")
            );
            assert_eq!(experts.len(), 4);
            assert_eq!(*routing_weight_multiplier, 128.0);
            for (role, matrix) in ["gate", "up", "down"].into_iter().zip(*shared) {
                assert_eq!(
                    plan.layout.modules[matrix].prefix,
                    format!("model.layers.{index}.mlp.shared_expert.{role}_proj")
                );
            }
            for (expert, matrices) in experts.iter().enumerate() {
                for (role, matrix) in ["gate", "up", "down"].into_iter().zip(*matrices) {
                    assert_eq!(
                        plan.layout.modules[matrix].prefix,
                        format!("model.layers.{index}.mlp.experts.{expert}.{role}_proj")
                    );
                }
            }
        }
    }
}
