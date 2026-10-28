use std::{collections::BTreeMap, fs, path::Path};

use crate::{
    EngError,
    memory::MemoryPolicy,
    weights::{nemotron_index::NemotronIndex, nemotron_shards::NemotronShardReader},
};

use super::{
    NemotronTextSpec,
    exec::{NemotronExecSpec, NemotronState},
    keys::{NemotronKeyClass, NemotronTensorPart},
};

/// Minimum budget: the policy share may tighten the ceiling above this
/// floor, never below it; the ceiling helper caps the floor at the
/// measured available bytes.
const AVAILABLE_RAM_FLOOR_BYTES: u64 = 20 * 1024 * 1024 * 1024;
pub(super) const ACTIVATION_WORKSPACE_BYTES: u64 = 256 * 1024 * 1024;

/// A bounded chunk's transport and projection storage, before placement.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChunkStreamPlan {
    pub(crate) banks: crate::expert_tier::StreamBankPlan,
    pub(crate) max_pairs: usize,
    pub(crate) up_shape: [usize; 2],
}

#[derive(Debug, Clone, Copy)]
struct MemoryComponents {
    dense_payload_bytes: u64,
    state_bytes: u64,
    expert_cache_bytes: u64,
    transient_margin_bytes: u64,
}

pub(super) fn admit_checkpoint(
    directory: &Path,
    descriptor: &NemotronTextSpec,
    spec: &NemotronExecSpec,
    capacity: u32,
    cache_entries: usize,
    policy: MemoryPolicy,
    available_bytes: u64,
) -> Result<(), EngError> {
    let payloads = checkpoint_payloads(directory, descriptor)?;
    let dense_payload_bytes = payloads.dense_payload_bytes;
    let largest_expert_entry_bytes = payloads.largest_expert_entry_bytes;
    let state_bytes = NemotronState::planned_bytes(spec, capacity)?;
    let cache_entries = u64::try_from(cache_entries)
        .map_err(|_| memory_error("expert cache entry count exceeds u64"))?;
    let expert_cache_bytes = cache_entries
        .checked_mul(largest_expert_entry_bytes)
        .ok_or_else(|| memory_error("expert cache bytes overflow u64"))?;
    // A cache miss peaks at one extra worst-entry payload: the loader documents a 2x-entry
    // decode peak. The fixed 256 MiB covers logits (~512 KiB at vocab 131072), attention
    // replay slices, and step vectors with wide headroom. This margin is not model-tuned.
    let transient_margin_bytes = largest_expert_entry_bytes
        .checked_add(ACTIVATION_WORKSPACE_BYTES)
        .ok_or_else(|| memory_error("transient margin bytes overflow u64"))?;

    check_memory(
        MemoryComponents {
            dense_payload_bytes,
            state_bytes,
            expert_cache_bytes,
            transient_margin_bytes,
        },
        policy,
        available_bytes,
    )
}

/// Decoded payload sizes the admission checks share, taken from one header scan.
#[derive(Debug, Clone, Copy)]
struct Payloads {
    dense_payload_bytes: u64,
    largest_dense_tensor_bytes: u64,
    largest_routed_entry_bytes: u64,
    largest_shared_entry_bytes: u64,
    largest_expert_entry_bytes: u64,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    cuda: CudaPayloads,
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[derive(Debug, Clone, Copy, Default)]
struct CudaPayloads {
    largest_packed_bytes: u64,
    largest_scale_bytes: u64,
    max_shape: [u64; 2],
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
impl CudaPayloads {
    fn observe_matrix(&mut self, [rows, columns]: [u64; 2]) {
        self.max_shape[0] = self.max_shape[0].max(rows);
        self.max_shape[1] = self.max_shape[1].max(columns);
    }
}

fn checkpoint_payloads(
    directory: &Path,
    descriptor: &NemotronTextSpec,
) -> Result<Payloads, EngError> {
    let index_path = directory.join("model.safetensors.index.json");
    let index_bytes = fs::read(&index_path)
        .map_err(|error| checkpoint_error(format!("read model.safetensors.index.json: {error}")))?;
    // NemotronIndex rejects duplicate JSON names before this second parse enumerates its keys.
    let index = NemotronIndex::from_json(&index_bytes)?;
    let root: serde_json::Value = serde_json::from_slice(&index_bytes)
        .map_err(|error| checkpoint_error(format!("parse index: {error}")))?;
    let weight_map = root
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| checkpoint_error("index has no weight_map object"))?;

    let mut names_by_shard: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for name in weight_map.keys() {
        if !name.starts_with("language_model.backbone.")
            && !name.starts_with("language_model.lm_head.")
        {
            continue;
        }
        let shard_name = index
            .shard_for_tensor(name)
            .ok_or_else(|| checkpoint_error(format!("index has no shard for tensor {name:?}")))?;
        names_by_shard.entry(shard_name).or_default().push(name);
    }

    let mut payloads = Payloads {
        dense_payload_bytes: 0,
        largest_dense_tensor_bytes: 0,
        largest_routed_entry_bytes: 0,
        largest_shared_entry_bytes: 0,
        largest_expert_entry_bytes: 0,
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        cuda: CudaPayloads::default(),
    };
    for (shard_name, names) in names_by_shard {
        let shard = NemotronShardReader::open(directory.join(shard_name))?;
        for name in names {
            let plan = descriptor.key_plan(name)?;
            let metadata = shard
                .tensor(name)
                .ok_or_else(|| checkpoint_error(format!("shard header has no tensor {name:?}")))?;
            if is_expert_class(plan.class) {
                #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                if plan.part == NemotronTensorPart::WeightScale {
                    payloads.cuda.largest_scale_bytes =
                        payloads.cuda.largest_scale_bytes.max(metadata.byte_len());
                }
                if plan.part != NemotronTensorPart::Weight {
                    continue;
                }
                let entry = expert_entry_payload_bytes(name, metadata.dtype(), metadata.shape())?;
                #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                {
                    let rows = metadata.shape()[0];
                    // The checked decoded entry has rows * columns * 4 bytes.
                    payloads.cuda.observe_matrix([rows, entry / rows / 4]);
                    payloads.cuda.largest_packed_bytes =
                        payloads.cuda.largest_packed_bytes.max(metadata.byte_len());
                }
                payloads.largest_expert_entry_bytes =
                    payloads.largest_expert_entry_bytes.max(entry);
                let class_max = match plan.class {
                    NemotronKeyClass::MoeExpertUp | NemotronKeyClass::MoeExpertDown => {
                        &mut payloads.largest_routed_entry_bytes
                    }
                    _ => &mut payloads.largest_shared_entry_bytes,
                };
                *class_max = (*class_max).max(entry);
            } else if matches!(
                plan.part,
                NemotronTensorPart::Weight | NemotronTensorPart::Bias | NemotronTensorPart::Value
            ) {
                let bytes = dense_tensor_payload_bytes(name, metadata.dtype(), metadata.shape())?;
                #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                if let &[rows, columns] = metadata.shape()
                    && rows > 0
                    && columns > 0
                {
                    payloads.cuda.observe_matrix([rows, columns]);
                }
                payloads.dense_payload_bytes = payloads
                    .dense_payload_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| memory_error("dense payload bytes overflow u64"))?;
                payloads.largest_dense_tensor_bytes =
                    payloads.largest_dense_tensor_bytes.max(bytes);
            }
        }
    }
    if payloads.largest_expert_entry_bytes == 0 {
        return Err(checkpoint_error(
            "index has no nonempty expert weight entries",
        ));
    }
    Ok(payloads)
}

fn is_expert_class(class: NemotronKeyClass) -> bool {
    matches!(
        class,
        NemotronKeyClass::MoeExpertUp
            | NemotronKeyClass::MoeExpertDown
            | NemotronKeyClass::MoeSharedUp
            | NemotronKeyClass::MoeSharedDown
    )
}

fn dense_tensor_payload_bytes(name: &str, dtype: &str, shape: &[u64]) -> Result<u64, EngError> {
    if !matches!(dtype, "BF16" | "F32" | "F8_E4M3") {
        return Err(checkpoint_error(format!(
            "dense tensor {name:?} has unsupported dtype {dtype}"
        )));
    }
    tensor_elements(name, shape)?
        .checked_mul(4)
        .ok_or_else(|| memory_error(format!("dense tensor {name:?} payload overflows u64")))
}

fn expert_entry_payload_bytes(name: &str, dtype: &str, shape: &[u64]) -> Result<u64, EngError> {
    let [rows, packed_columns] = shape else {
        return Err(checkpoint_error(format!(
            "expert tensor {name:?} must be a rank-two matrix"
        )));
    };
    if dtype != "U8" || *rows == 0 || *packed_columns == 0 {
        return Err(checkpoint_error(format!(
            "expert tensor {name:?} must be a nonempty U8 matrix"
        )));
    }
    let columns = packed_columns
        .checked_mul(2)
        .ok_or_else(|| memory_error(format!("expert tensor {name:?} columns overflow u64")))?;
    if !columns.is_multiple_of(16) {
        return Err(checkpoint_error(format!(
            "expert tensor {name:?} columns must be a multiple of 16"
        )));
    }
    rows.checked_mul(columns)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| memory_error(format!("expert tensor {name:?} payload overflows u64")))
}

fn tensor_elements(name: &str, shape: &[u64]) -> Result<u64, EngError> {
    shape.iter().try_fold(1u64, |elements, &dimension| {
        elements
            .checked_mul(dimension)
            .ok_or_else(|| memory_error(format!("tensor {name:?} element count overflows u64")))
    })
}

/// The effective ceiling: the larger of the absolute floor and the policy
/// share of the measured headroom, capped at that headroom. The min cap
/// exists because the floor alone would otherwise admit a 20 GiB plan at
/// 10 GiB of measured headroom and commit twice what is actually free.
fn admission_ceiling(policy: MemoryPolicy, available_bytes: u64) -> u64 {
    available_bytes.min(policy.limit(available_bytes).max(AVAILABLE_RAM_FLOOR_BYTES))
}

fn check_memory(
    components: MemoryComponents,
    policy: MemoryPolicy,
    available_bytes: u64,
) -> Result<(), EngError> {
    let ceiling = admission_ceiling(policy, available_bytes);
    let mut planned = 0u64;
    for (name, bytes) in [
        ("dense payload", components.dense_payload_bytes),
        ("state", components.state_bytes),
        ("expert cache", components.expert_cache_bytes),
        ("transient margin", components.transient_margin_bytes),
    ] {
        planned = planned
            .checked_add(bytes)
            .ok_or_else(|| memory_error(format!("{name} bytes overflow total planned bytes")))?;
        if planned > ceiling {
            return Err(memory_error(format!(
                "{name} exceeds the memory ceiling: planned {planned} bytes, ceiling {ceiling} bytes"
            )));
        }
    }
    Ok(())
}

/// Bytes of the routed and shared experts one MoE layer keeps outside the cache.
fn retained_selected_bytes(routed_pairs: u64, routed_max: u64, shared_max: u64) -> Option<u64> {
    routed_pairs.checked_mul(routed_max).and_then(|routed| {
        shared_max
            .checked_mul(2)
            .and_then(|shared| routed.checked_add(shared))
    })
}

#[derive(Debug, Clone, Copy)]
struct ServingComponents {
    dense_payload_bytes: u64,
    state_bytes: u64,
    expert_cache_bytes: u64,
    retained_experts_bytes: u64,
    dense_copy_bytes: u64,
    attention_scratch_bytes: u64,
    miss_peak_bytes: u64,
    fixed_workspace_bytes: u64,
}

fn check_serving_memory(
    components: ServingComponents,
    policy: MemoryPolicy,
    available_bytes: u64,
) -> Result<u64, EngError> {
    let ceiling = admission_ceiling(policy, available_bytes);
    let mut planned = 0u64;
    for (name, bytes) in [
        ("dense payload", components.dense_payload_bytes),
        ("three state payloads", components.state_bytes),
        ("expert cache", components.expert_cache_bytes),
        (
            "retained selected experts",
            components.retained_experts_bytes,
        ),
        ("dense copies", components.dense_copy_bytes),
        (
            "attention replay scratch",
            components.attention_scratch_bytes,
        ),
        ("expert miss peak", components.miss_peak_bytes),
        ("fixed workspace", components.fixed_workspace_bytes),
    ] {
        planned = planned
            .checked_add(bytes)
            .ok_or_else(|| memory_error(format!("{name} bytes overflow total planned bytes")))?;
        if planned > ceiling {
            return Err(memory_error(format!(
                "{name} exceeds the memory ceiling: planned {planned} bytes, ceiling {ceiling} bytes"
            )));
        }
    }
    Ok(planned)
}

/// Serving admission for one CPU-baseline engine start.
///
/// The plan covers what the stack holds at once: the dense payload, the pristine,
/// live and replacement states, the expert cache, the selected experts the caller
/// keeps outside the cache, two dense copies, attention replay scratch that grows
/// with capacity, one expert miss peak, and a fixed workspace.
/// The ceiling is the larger of the 20 GiB floor and the policy share of
/// the available bytes, capped at the available bytes.
pub(super) fn admit_serving(
    directory: &Path,
    descriptor: &NemotronTextSpec,
    spec: &NemotronExecSpec,
    capacity: u32,
    cache_entries: usize,
    policy: MemoryPolicy,
    available_bytes: u64,
) -> Result<(), EngError> {
    let (components, _) = serving_plan(directory, descriptor, spec, capacity, cache_entries)?;
    check_serving_memory(components, policy, available_bytes).map(|_| ())
}

/// One header scan supplies both the retained serving plan and its tensor maxima.
fn serving_plan(
    directory: &Path,
    descriptor: &NemotronTextSpec,
    spec: &NemotronExecSpec,
    capacity: u32,
    cache_entries: usize,
) -> Result<(ServingComponents, Payloads), EngError> {
    let state_bytes = NemotronState::planned_bytes(spec, capacity)?;
    let payloads = checkpoint_payloads(directory, descriptor)?;
    if payloads.largest_routed_entry_bytes == 0 || payloads.largest_shared_entry_bytes == 0 {
        return Err(checkpoint_error(
            "serving admission needs routed and shared expert entry maxima",
        ));
    }
    let hidden = u64::from(descriptor.geometry().hidden_size);
    let capacity = u64::from(capacity);
    let cache_entries = u64::try_from(cache_entries)
        .map_err(|_| memory_error("expert cache entry count exceeds u64"))?;
    let routed_pairs = u64::from(descriptor.moe_geometry().experts_per_token)
        .checked_mul(2)
        .ok_or_else(|| memory_error("routed retention pairs overflow u64"))?;
    let three_states = state_bytes
        .checked_mul(3)
        .ok_or_else(|| memory_error("three state payloads overflow u64"))?;
    let expert_cache = cache_entries
        .checked_mul(payloads.largest_expert_entry_bytes)
        .ok_or_else(|| memory_error("expert cache bytes overflow u64"))?;
    let retained = retained_selected_bytes(
        routed_pairs,
        payloads.largest_routed_entry_bytes,
        payloads.largest_shared_entry_bytes,
    )
    .ok_or_else(|| memory_error("retained selected experts overflow u64"))?;
    let dense_copies = payloads
        .largest_dense_tensor_bytes
        .checked_mul(2)
        .ok_or_else(|| memory_error("dense copies overflow u64"))?;
    let attention = capacity
        .checked_mul(hidden)
        .and_then(|elements| elements.checked_mul(4))
        .and_then(|bytes| bytes.checked_mul(12))
        .ok_or_else(|| memory_error("attention replay scratch overflow u64"))?;
    Ok((
        ServingComponents {
            dense_payload_bytes: payloads.dense_payload_bytes,
            state_bytes: three_states,
            expert_cache_bytes: expert_cache,
            retained_experts_bytes: retained,
            dense_copy_bytes: dense_copies,
            attention_scratch_bytes: attention,
            miss_peak_bytes: payloads.largest_expert_entry_bytes,
            fixed_workspace_bytes: ACTIVATION_WORKSPACE_BYTES,
        },
        payloads,
    ))
}

pub(super) fn host_available_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|contents| parse_mem_available(&contents))
    }
    #[cfg(not(target_os = "linux"))]
    None
}

#[cfg(target_os = "linux")]
fn parse_mem_available(contents: &str) -> Option<u64> {
    let line = contents
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))?;
    let mut fields = line.strip_prefix("MemAvailable:")?.split_whitespace();
    let kib = fields.next()?.parse::<u64>().ok()?;
    (fields.next()? == "kB").then_some(kib.checked_mul(1024)?)
}

fn memory_error(message: impl Into<String>) -> EngError {
    EngError::MemoryPlan(format!("Nemotron admission: {}", message.into()))
}

fn checkpoint_error(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("Nemotron admission: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::{MemoryComponents, admission_ceiling, check_memory};
    use crate::{EngError, memory::MemoryPolicy};

    const GIB: u64 = 1024 * 1024 * 1024;
    const FLOOR: u64 = 20 * GIB;

    fn plan(total_bytes: u64) -> MemoryComponents {
        let quarter = total_bytes / 4;
        MemoryComponents {
            dense_payload_bytes: quarter,
            state_bytes: quarter,
            expert_cache_bytes: quarter,
            transient_margin_bytes: total_bytes - 3 * quarter,
        }
    }

    #[test]
    fn low_headroom_caps_the_floor_at_the_measured_available_bytes() {
        let available = 10 * GIB;
        let default = MemoryPolicy::default_share();
        assert!(default.limit(available) < FLOOR);
        assert_eq!(admission_ceiling(default, available), available);

        // The floor lifts the 8.5 GiB share, so a 9 GiB plan stays admissible.
        check_memory(plan(9 * GIB), default, available)
            .expect("the floor widens the ceiling above the share");
        // The cap holds: a 15 GiB plan exceeds the measured 10 GiB headroom.
        let error = check_memory(plan(15 * GIB), default, available)
            .expect_err("the headroom caps the floor");
        let EngError::MemoryPlan(message) = error else {
            panic!("expected a memory-plan rejection, got {error:?}");
        };
        assert!(message.contains("expert cache"), "{message}");
    }

    #[test]
    fn a_twenty_gib_plan_is_refused_at_ten_gib_of_headroom() {
        let error = check_memory(plan(20 * GIB), MemoryPolicy::default_share(), 10 * GIB)
            .expect_err("the floor cannot exceed the measured headroom");
        assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
    }

    #[test]
    fn far_above_the_floor_the_share_decides_and_tightens_below_the_headroom() {
        let available = 200 * GIB;
        let policy = MemoryPolicy::default_share();
        assert_eq!(admission_ceiling(policy, available), 170 * GIB);
        assert!(
            policy.limit(available) > FLOOR,
            "the share, not the floor, supplies the ceiling"
        );

        check_memory(plan(165 * GIB), policy, available)
            .expect("the plan fits under the share ceiling");
        let error = check_memory(plan(171 * GIB), policy, available)
            .expect_err("the share refuses above its own ceiling");
        let EngError::MemoryPlan(message) = error else {
            panic!("expected a memory-plan rejection, got {error:?}");
        };
        assert!(message.contains("transient margin"), "{message}");
    }

    #[test]
    fn a_half_share_refuses_earlier_than_the_default_share() {
        let available = 200 * GIB;
        let total = 120 * GIB;
        check_memory(plan(total), MemoryPolicy::default_share(), available)
            .expect("the 170 GiB default-share ceiling admits this plan");
        let error = check_memory(
            plan(total),
            MemoryPolicy::from_permille(500).expect("valid share"),
            available,
        )
        .expect_err("the 100 GiB half-share ceiling refuses this plan");
        assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
    }

    // CPU entrypoint regression tests: user-approved bounded fixtures through real readers.
    mod cpu_entrypoints {
        use std::{collections::BTreeMap, fs, path::Path};

        use serde_json::{Value, json};
        use tempfile::TempDir;

        use super::super::{
            ACTIVATION_WORKSPACE_BYTES, admit_checkpoint, admit_serving, checkpoint_payloads,
            serving_plan,
        };
        use crate::{
            EngError,
            family::nemotron35::NemotronTextSpec,
            family::nemotron35::exec::{NemotronExecSpec, NemotronState},
            memory::MemoryPolicy,
        };

        const NORM: &str = "language_model.backbone.norm_f.weight";
        const PROJ: &str = "language_model.backbone.layers.0.mixer.in_proj.weight";
        const EXPERT: &str = "language_model.backbone.layers.1.mixer.experts.0";
        const SHARED: &str = "language_model.backbone.layers.1.mixer.shared_experts";

        struct Shard {
            header: BTreeMap<String, Value>,
            bytes: Vec<u8>,
        }

        impl Shard {
            fn new() -> Self {
                Self {
                    header: BTreeMap::new(),
                    bytes: Vec::new(),
                }
            }

            fn add(&mut self, name: &str, dtype: &str, shape: &[u32], bytes: &[u8]) {
                let start = self.bytes.len();
                self.bytes.extend_from_slice(bytes);
                self.header.insert(
                    name.into(),
                    json!({"dtype": dtype, "shape": shape, "data_offsets": [start, self.bytes.len()]}),
                );
            }

            fn write(&self, dir: &Path, name: &str, index: &mut BTreeMap<String, String>) {
                let header = serde_json::to_vec(&self.header).unwrap();
                let mut file = (header.len() as u64).to_le_bytes().to_vec();
                file.extend(header);
                file.extend(&self.bytes);
                fs::write(dir.join(name), file).unwrap();
                for key in self.header.keys() {
                    index.insert(key.clone(), name.into());
                }
            }
        }

        fn fixture() -> (TempDir, NemotronTextSpec, NemotronExecSpec) {
            let dir = tempfile::tempdir().unwrap();
            fs::write(
                dir.path().join("config.json"),
                include_bytes!("../../../tests/fixtures/nemotron35/config.json"),
            )
            .unwrap();
            let mut dense = Shard::new();
            dense.add(NORM, "BF16", &[2], &[0x80, 0x3f, 0x00, 0xc0]);
            dense.add(PROJ, "F8_E4M3", &[1, 4], &[0x38, 0x40, 0xb8, 0x30]);
            dense.add(
                &PROJ.replace(".weight", ".weight_scale"),
                "F32",
                &[1],
                &2.0f32.to_le_bytes(),
            );
            dense.add(
                &PROJ.replace(".weight", ".input_scale"),
                "F32",
                &[],
                &0.25f32.to_le_bytes(),
            );
            let mut experts = Shard::new();
            for base in [EXPERT, SHARED] {
                for which in ["up_proj", "down_proj"] {
                    let base = format!("{base}.{which}");
                    experts.add(&format!("{base}.weight"), "U8", &[1, 8], &[0u8; 8]);
                    experts.add(&format!("{base}.weight_scale"), "F8_E4M3", &[1, 1], &[0x40]);
                    experts.add(
                        &format!("{base}.weight_scale_2"),
                        "F32",
                        &[],
                        &0.5f32.to_le_bytes(),
                    );
                    experts.add(
                        &format!("{base}.input_scale"),
                        "F32",
                        &[1],
                        &0.125f32.to_le_bytes(),
                    );
                }
            }
            let mut index = BTreeMap::new();
            dense.write(dir.path(), "dense.safetensors", &mut index);
            experts.write(dir.path(), "experts.safetensors", &mut index);
            fs::write(
                dir.path().join("model.safetensors.index.json"),
                serde_json::to_vec(&json!({"weight_map": index})).unwrap(),
            )
            .unwrap();
            let config = fs::read(dir.path().join("config.json")).unwrap();
            let descriptor = NemotronTextSpec::from_config(&config).unwrap();
            let spec = NemotronExecSpec::from_descriptor(&descriptor).unwrap();
            (dir, descriptor, spec)
        }

        fn checkpoint_total(
            dir: &Path,
            descriptor: &NemotronTextSpec,
            spec: &NemotronExecSpec,
        ) -> u64 {
            let payloads = checkpoint_payloads(dir, descriptor).unwrap();
            let state = NemotronState::planned_bytes(spec, 16).unwrap();
            let cache = 4u64 * payloads.largest_expert_entry_bytes;
            let transient = payloads.largest_expert_entry_bytes + ACTIVATION_WORKSPACE_BYTES;
            payloads.dense_payload_bytes + state + cache + transient
        }

        #[test]
        fn admit_checkpoint_accepts_the_exact_total_and_refuses_one_byte_less() {
            let (dir, descriptor, spec) = fixture();
            let total = checkpoint_total(dir.path(), &descriptor, &spec);
            assert!(total > 0 && total < 20 * 1024 * 1024 * 1024);
            admit_checkpoint(
                dir.path(),
                &descriptor,
                &spec,
                16,
                4,
                MemoryPolicy::default_share(),
                total,
            )
            .expect("the exact total fits the measured ceiling");
            let error = admit_checkpoint(
                dir.path(),
                &descriptor,
                &spec,
                16,
                4,
                MemoryPolicy::default_share(),
                total - 1,
            )
            .expect_err("one byte below the total exceeds the ceiling");
            assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
        }

        #[test]
        fn admit_checkpoint_reports_checkpoint_errors_for_a_missing_index() {
            let (dir, descriptor, spec) = fixture();
            fs::remove_file(dir.path().join("model.safetensors.index.json")).unwrap();
            let error = admit_checkpoint(
                dir.path(),
                &descriptor,
                &spec,
                16,
                4,
                MemoryPolicy::default_share(),
                u64::MAX,
            )
            .expect_err("a missing index cannot admit a checkpoint");
            let EngError::Checkpoint(message) = &error else {
                panic!("expected a checkpoint rejection, got {error:?}");
            };
            assert!(
                message.contains("model.safetensors.index.json"),
                "{message}"
            );
        }

        #[test]
        fn admit_checkpoint_refuses_overflowing_cache_entries() {
            let (dir, descriptor, spec) = fixture();
            let error = admit_checkpoint(
                dir.path(),
                &descriptor,
                &spec,
                16,
                usize::MAX,
                MemoryPolicy::default_share(),
                u64::MAX,
            )
            .expect_err("the expert cache product must refuse overflow");
            assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
        }

        #[test]
        fn admit_serving_accepts_the_exact_total_and_refuses_one_byte_less() {
            let (dir, descriptor, spec) = fixture();
            let (components, _) = serving_plan(dir.path(), &descriptor, &spec, 16, 4).unwrap();
            let total = components.dense_payload_bytes
                + components.state_bytes
                + components.expert_cache_bytes
                + components.retained_experts_bytes
                + components.dense_copy_bytes
                + components.attention_scratch_bytes
                + components.miss_peak_bytes
                + components.fixed_workspace_bytes;
            assert!(total > 0 && total < 20 * 1024 * 1024 * 1024);
            admit_serving(
                dir.path(),
                &descriptor,
                &spec,
                16,
                4,
                MemoryPolicy::default_share(),
                total,
            )
            .expect("the exact serving total fits the measured ceiling");
            let error = admit_serving(
                dir.path(),
                &descriptor,
                &spec,
                16,
                4,
                MemoryPolicy::default_share(),
                total - 1,
            )
            .expect_err("one byte below the serving total exceeds the ceiling");
            assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
        }
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(super) mod cuda {
    use std::sync::Mutex;

    use super::{
        NemotronExecSpec, NemotronTextSpec, admission_ceiling, check_serving_memory, memory_error,
        serving_plan,
    };
    use crate::{
        EngError,
        kernels::cuda::{Admission, CudaContext, MemoryInfo, PINNED_STAGE_LIMIT_BYTES},
        memory::MemoryPolicy,
    };

    /// Requested ceiling for the device-resident weight cache. Admission caps
    /// the request by host headroom, because every cached device weight also
    /// retains its decoded host tensor for key stability.
    pub(crate) const DEVICE_WEIGHT_CACHE_REQUEST_BYTES: u64 = 12 * 1024 * 1024 * 1024;

    /// Bounded planning data. Only the reservation function can issue its capability.
    pub(crate) struct CudaPlanInput {
        pub(crate) host_base_bytes: u64,
        pub(crate) largest_tensor_bytes: u64,
        pub(crate) packed_pair_bytes: u64,
        pub(crate) max_shape: [u64; 2],
        pub(crate) device_weight_cache_bytes: u64,
    }

    #[derive(Debug, Clone, Copy)]
    pub(crate) struct CudaTransferLimits {
        max_shape: [usize; 2],
        pageable: u64,
        packed_pair: u64,
    }

    impl CudaTransferLimits {
        pub(crate) fn check_shape(self, [rows, columns]: [usize; 2]) -> Result<(), EngError> {
            if rows == 0 || columns == 0 {
                return Err(EngError::BadRequest(
                    "CUDA projection dimensions must be positive".to_owned(),
                ));
            }
            if rows > self.max_shape[0] || columns > self.max_shape[1] {
                return Err(memory_error(format!(
                    "CUDA projection shape [{rows}, {columns}] exceeds admitted maximum {:?}",
                    self.max_shape
                )));
            }
            Ok(())
        }

        pub(crate) fn check_pageable(self, bytes: usize) -> Result<(), EngError> {
            let bytes = u64::try_from(bytes)
                .map_err(|_| memory_error("pageable transfer bytes exceed u64"))?;
            self.check(bytes, self.pageable, "pageable transfer")
        }

        pub(crate) fn check_packed(self, packed: usize, scale: usize) -> Result<(), EngError> {
            let bytes = packed
                .checked_add(scale)
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| memory_error("packed transfer bytes overflow"))?;
            self.check(bytes, self.packed_pair, "packed transfer")
        }

        fn check(self, bytes: u64, maximum: u64, kind: &str) -> Result<(), EngError> {
            if bytes > maximum {
                return Err(memory_error(format!(
                    "{kind} needs {bytes} bytes, admitted maximum {maximum} bytes"
                )));
            }
            Ok(())
        }
    }

    #[derive(Debug)]
    struct CudaPlan {
        host_total: u64,
        pinned_peak: u64,
        device_peak: u64,
        weight_cache: u64,
        max_payload: usize,
        workspace: Admission,
        transfers: CudaTransferLimits,
    }

    impl CudaPlan {
        fn new(input: CudaPlanInput, cache_ceiling: u64) -> Result<Self, EngError> {
            let limit = u64::try_from(PINNED_STAGE_LIMIT_BYTES)
                .map_err(|_| memory_error("stage limit exceeds u64"))?;
            let max_payload = usize::try_from(input.largest_tensor_bytes.min(limit))
                .map_err(|_| memory_error("staging payload exceeds host limits"))?;
            if max_payload == 0 {
                return Err(memory_error(
                    "CUDA admission needs a nonempty matrix payload",
                ));
            }
            let rows = usize::try_from(input.max_shape[0])
                .map_err(|_| memory_error("projection rows exceed host limits"))?;
            let columns = usize::try_from(input.max_shape[1])
                .map_err(|_| memory_error("projection columns exceed host limits"))?;
            let (workspace, work_bytes) = CudaContext::fixed_projection_plan([rows, columns])?;
            let pinned_peak = u64::try_from(max_payload)
                .ok()
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or_else(|| memory_error("staging replacement envelope overflows u64"))?;
            let pageable = if input.largest_tensor_bytes > limit {
                input.largest_tensor_bytes
            } else {
                0
            };
            let host_base_total = input
                .host_base_bytes
                .checked_add(pinned_peak)
                .ok_or_else(|| memory_error("CUDA host plan overflows u64"))?;
            let weight_cache = input
                .device_weight_cache_bytes
                .min(cache_ceiling.saturating_sub(host_base_total));
            // Caching uploads each weight once through the pageable path, so
            // every single-weight upload must be admissible when caching is on.
            let pageable = if weight_cache > 0 {
                pageable.max(input.largest_tensor_bytes)
            } else {
                pageable
            };
            let host_total = host_base_total
                .checked_add(weight_cache)
                .ok_or_else(|| memory_error("CUDA host plan overflows u64"))?;
            // These terms are additive even when their maxima do not coincide.
            let device_peak = pinned_peak
                .checked_add(work_bytes)
                .and_then(|bytes| bytes.checked_add(pageable))
                .and_then(|bytes| bytes.checked_add(input.packed_pair_bytes))
                .and_then(|bytes| bytes.checked_add(weight_cache))
                .ok_or_else(|| memory_error("CUDA device plan overflows u64"))?;
            Ok(Self {
                host_total,
                pinned_peak,
                device_peak,
                weight_cache,
                max_payload,
                workspace,
                transfers: CudaTransferLimits {
                    max_shape: [rows, columns],
                    pageable,
                    packed_pair: input.packed_pair_bytes,
                },
            })
        }
    }

    struct Reservations {
        host: u64,
        device: u64,
        failed: bool,
    }

    static HELD: Mutex<Reservations> = Mutex::new(Reservations {
        host: 0,
        device: 0,
        failed: false,
    });

    /// Ordinary destruction retains credit. Only proven cleanup returns native credit.
    #[derive(Debug)]
    pub(crate) struct CudaAdmission {
        plan: CudaPlan,
        policy: MemoryPolicy,
        available_host: u64,
        charged_device: u64,
        shared_device_host: u64,
        stream_plan: Option<super::ChunkStreamPlan>,
    }

    pub(crate) fn admit_cuda_plan(
        input: CudaPlanInput,
        policy: MemoryPolicy,
        available_host: u64,
    ) -> Result<CudaAdmission, EngError> {
        let ceiling = admission_ceiling(policy, available_host);
        let plan = CudaPlan::new(input, ceiling)?;
        let mut held = HELD
            .lock()
            .map_err(|_| memory_error("CUDA reservation ledger is poisoned"))?;
        if held.failed {
            return Err(memory_error("CUDA reservation ledger is unavailable"));
        }
        let host = held
            .host
            .checked_add(plan.host_total)
            .ok_or_else(|| memory_error("aggregate CUDA host reservation overflows u64"))?;
        if host > ceiling {
            return Err(memory_error(format!(
                "aggregate CUDA host reservation needs {host} bytes, ceiling {ceiling} bytes"
            )));
        }
        held.host = host;
        Ok(CudaAdmission {
            plan,
            policy,
            available_host,
            charged_device: 0,
            shared_device_host: 0,
            stream_plan: None,
        })
    }

    impl CudaAdmission {
        pub(crate) fn max_payload(&self) -> usize {
            self.plan.max_payload
        }

        pub(crate) fn workspace(&self) -> &Admission {
            &self.plan.workspace
        }

        pub(crate) fn transfer_limits(&self) -> CudaTransferLimits {
            self.plan.transfers
        }

        pub(crate) fn device_weight_cache_bytes(&self) -> u64 {
            self.plan.weight_cache
        }

        pub(crate) fn reserve_streaming(
            &mut self,
            plan: super::ChunkStreamPlan,
        ) -> Result<(), EngError> {
            if self.stream_plan.is_some() {
                return Err(memory_error("chunk streaming is already reserved"));
            }
            // Banks partition optional weight caching instead of overcommitting it.
            let cache = self
                .plan
                .weight_cache
                .min(plan.banks.host_bytes().max(plan.banks.device_bytes()));
            let host_total = self
                .plan
                .host_total
                .checked_sub(cache)
                .and_then(|bytes| bytes.checked_add(plan.banks.host_bytes()))
                .ok_or_else(|| memory_error("stream host reservation overflows u64"))?;
            let device_peak = self
                .plan
                .device_peak
                .checked_sub(cache)
                .and_then(|bytes| bytes.checked_add(plan.banks.device_bytes()))
                .ok_or_else(|| memory_error("stream device reservation overflows u64"))?;
            let mut held = HELD
                .lock()
                .map_err(|_| memory_error("CUDA reservation ledger is poisoned"))?;
            if held.failed {
                return Err(memory_error("CUDA reservation ledger is unavailable"));
            }
            let host = held
                .host
                .checked_sub(self.plan.host_total)
                .and_then(|bytes| bytes.checked_add(host_total))
                .ok_or_else(|| memory_error("aggregate stream host reservation overflows u64"))?;
            if host > admission_ceiling(self.policy, self.available_host) {
                return Err(memory_error("chunk streaming exceeds the host ceiling"));
            }
            held.host = host;
            self.plan.host_total = host_total;
            self.plan.device_peak = device_peak;
            self.plan.weight_cache -= cache;
            self.stream_plan = Some(plan);
            Ok(())
        }

        pub(crate) fn stream_plan(&self) -> Option<super::ChunkStreamPlan> {
            self.stream_plan
        }

        /// The backend calls this under dispatch, before any staged allocation.
        pub(crate) fn into_cleanup(
            mut self,
            memory: MemoryInfo,
            integrated: bool,
        ) -> Result<Box<dyn FnOnce()>, EngError> {
            if memory.free > memory.total {
                return Err(memory_error("CUDA free memory exceeds reported total"));
            }
            let free = u64::try_from(memory.free)
                .map_err(|_| memory_error("CUDA free memory exceeds u64"))?;
            let shared = if integrated { self.plan.device_peak } else { 0 };
            {
                let mut held = HELD
                    .lock()
                    .map_err(|_| memory_error("CUDA reservation ledger is poisoned"))?;
                if held.failed {
                    return Err(memory_error("CUDA reservation ledger is unavailable"));
                }
                let host = held
                    .host
                    .checked_add(shared)
                    .ok_or_else(|| memory_error("shared CUDA host reservation overflows u64"))?;
                let device = held
                    .device
                    .checked_add(self.plan.device_peak)
                    .ok_or_else(|| {
                        memory_error("aggregate CUDA device reservation overflows u64")
                    })?;
                if host > admission_ceiling(self.policy, self.available_host) {
                    return Err(memory_error(
                        "shared CUDA device reservation exceeds host ceiling",
                    ));
                }
                if device > self.policy.limit(free) {
                    return Err(memory_error(format!(
                        "aggregate CUDA device reservation needs {device} bytes, ceiling {} bytes",
                        self.policy.limit(free)
                    )));
                }
                held.host = host;
                held.device = device;
                self.charged_device = self.plan.device_peak;
                self.shared_device_host = shared;
            }
            Ok(Box::new(move || self.release_native_after_cleanup()))
        }

        fn release_native_after_cleanup(self) {
            let Ok(mut held) = HELD.lock() else {
                return;
            };
            if held.failed {
                return;
            }
            let stream_host = self.stream_plan.map_or(0, |plan| plan.banks.host_bytes());
            let refund_host = self
                .plan
                .pinned_peak
                .checked_add(stream_host)
                .and_then(|bytes| bytes.checked_add(self.shared_device_host));
            let host = refund_host.and_then(|bytes| held.host.checked_sub(bytes));
            let device = held.device.checked_sub(self.charged_device);
            let (Some(host), Some(device)) = (host, device) else {
                held.failed = true;
                return;
            };
            held.host = host;
            held.device = device;
            // shortcut: CPU baseline credit stays held, release it when exported tensor aliases have a proven lifetime.
        }
    }

    /// Both production CUDA entry points use this complete retained serving plan.
    pub(crate) fn admit_cuda_serving(
        directory: &std::path::Path,
        descriptor: &NemotronTextSpec,
        spec: &NemotronExecSpec,
        capacity: u32,
        cache_entries: usize,
        policy: MemoryPolicy,
        available_host: u64,
    ) -> Result<CudaAdmission, EngError> {
        let (components, payloads) =
            serving_plan(directory, descriptor, spec, capacity, cache_entries)?;
        let host_base_bytes = check_serving_memory(components, policy, available_host)?;
        let packed_pair_bytes = payloads
            .cuda
            .largest_packed_bytes
            .checked_add(payloads.cuda.largest_scale_bytes)
            .ok_or_else(|| memory_error("packed weight/scale peak overflows u64"))?;
        admit_cuda_plan(
            CudaPlanInput {
                host_base_bytes,
                largest_tensor_bytes: payloads
                    .largest_dense_tensor_bytes
                    .max(payloads.largest_expert_entry_bytes),
                packed_pair_bytes,
                max_shape: payloads.cuda.max_shape,
                device_weight_cache_bytes: DEVICE_WEIGHT_CACHE_REQUEST_BYTES,
            },
            policy,
            available_host,
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::{process::Command, sync::Barrier};

        const CASE: &str = "INFERENCE_TEST_NEM35_RESERVATION_CASE";
        const NAME: &str = "family::nemotron35::admission::cuda::tests::owned_reservation_contract";

        fn small(base: u64) -> CudaPlanInput {
            CudaPlanInput {
                host_base_bytes: base,
                largest_tensor_bytes: 64,
                packed_pair_bytes: 0,
                max_shape: [1, 16],
                device_weight_cache_bytes: 0,
            }
        }

        #[test]
        fn weight_cache_request_is_capped_by_host_headroom() {
            let mut input = small(100);
            input.device_weight_cache_bytes = u64::MAX;
            let ceiling = admission_ceiling(MemoryPolicy::default_share(), 10_000);
            let plan = CudaPlan::new(input, ceiling).unwrap();
            // small() gives pinned_peak = 2 * 64 = 128, so the base plan is 228
            // and the cache request collapses to the remaining headroom.
            assert_eq!(plan.host_total, ceiling);
            assert_eq!(plan.weight_cache, ceiling - 228);
            assert_eq!(plan.transfers.pageable, 64);
            // A zero request keeps the legacy plan byte for byte, and the cache
            // is the only device-peak difference between the two plans.
            let legacy = CudaPlan::new(small(100), ceiling).unwrap();
            assert_eq!(legacy.weight_cache, 0);
            assert_eq!(legacy.host_total, 228);
            assert_eq!(legacy.transfers.pageable, 0);
            // Cache mode also admits single-weight pageable uploads, so the
            // device-peak delta is the cache plus the raised pageable term.
            assert_eq!(
                plan.device_peak - legacy.device_peak,
                plan.weight_cache + 64
            );
        }

        fn reserve(base: u64, available: u64) -> Result<CudaAdmission, EngError> {
            admit_cuda_plan(small(base), MemoryPolicy::default_share(), available)
        }

        fn memory(free: u64) -> MemoryInfo {
            let free = usize::try_from(free).unwrap();
            MemoryInfo { free, total: free }
        }

        fn held() -> (u64, u64) {
            let held = HELD.lock().unwrap_or_else(|error| error.into_inner());
            (held.host, held.device)
        }

        fn reservation_snapshot(ticket: &CudaAdmission) -> (String, (u64, u64, bool)) {
            let held = HELD.lock().unwrap_or_else(|error| error.into_inner());
            (format!("{ticket:?}"), (held.host, held.device, held.failed))
        }

        fn host_boundaries() {
            assert!(matches!(reserve(100, 227), Err(EngError::MemoryPlan(_))));
            assert_eq!(held(), (0, 0));
            let ticket = reserve(100, 228).unwrap();
            assert_eq!(held(), (228, 0));
            assert!(matches!(reserve(0, 228), Err(EngError::MemoryPlan(_))));
            assert_eq!(held(), (228, 0));
            drop(ticket);
            assert_eq!(held(), (228, 0), "ordinary Drop cannot return credit");
        }

        fn cleanup_and_reload() {
            let ticket = reserve(100, 1_000).unwrap();
            let peak = ticket.plan.device_peak;
            let cleanup = ticket.into_cleanup(memory(peak * 2), false).unwrap();
            assert_eq!(held(), (228, peak));
            cleanup();
            assert_eq!(held(), (100, 0), "CPU aliases retain their baseline");
            assert!(matches!(reserve(100, 327), Err(EngError::MemoryPlan(_))));
            let second = reserve(100, 328).unwrap();
            let cleanup = second.into_cleanup(memory(peak * 2), false).unwrap();
            cleanup();
            assert_eq!(held(), (200, 0));
        }

        fn discarded_obligation() {
            let ticket = reserve(100, 1_000).unwrap();
            let peak = ticket.plan.device_peak;
            let cleanup = ticket.into_cleanup(memory(peak * 2), false).unwrap();
            drop(cleanup);
            assert_eq!(held(), (228, peak));
            let second = reserve(100, 1_000).unwrap();
            assert!(matches!(
                second.into_cleanup(memory(peak * 2), false),
                Err(EngError::MemoryPlan(_))
            ));
            assert_eq!(held(), (456, peak), "failed activation retains host credit");
        }

        fn integrated_atomicity() {
            let peak = CudaPlan::new(small(100), u64::MAX).unwrap().device_peak;
            let first = reserve(100, 228 + peak - 1).unwrap();
            assert!(matches!(
                first.into_cleanup(memory(peak * 2), true),
                Err(EngError::MemoryPlan(_))
            ));
            assert_eq!(
                held(),
                (228, 0),
                "refusal must not charge either extra tier"
            );
            let second = reserve(100, 456 + peak).unwrap();
            let cleanup = second.into_cleanup(memory(peak * 2), true).unwrap();
            assert_eq!(held(), (456 + peak, peak));
            cleanup();
            assert_eq!(held(), (328, 0));
        }

        fn concurrent_same_snapshot() {
            let barrier = std::sync::Arc::new(Barrier::new(16));
            let workers: Vec<_> = (0..16)
                .map(|_| {
                    let barrier = barrier.clone();
                    std::thread::spawn(move || {
                        barrier.wait();
                        match reserve(100, 912) {
                            Ok(ticket) => {
                                drop(ticket);
                                true
                            }
                            Err(EngError::MemoryPlan(_)) => false,
                            Err(error) => panic!("unexpected admission error: {error:?}"),
                        }
                    })
                })
                .collect();
            let accepted = workers
                .into_iter()
                .map(|worker| usize::from(worker.join().unwrap()))
                .sum::<usize>();
            assert_eq!(accepted, 4);
            assert_eq!(held(), (912, 0));
        }

        fn invalid_and_overflow() {
            for input in [
                CudaPlanInput {
                    max_shape: [0, 16],
                    ..small(0)
                },
                CudaPlanInput {
                    max_shape: [1, 0],
                    ..small(0)
                },
                small(u64::MAX),
                CudaPlanInput {
                    largest_tensor_bytes: 0,
                    ..small(0)
                },
                CudaPlanInput {
                    packed_pair_bytes: u64::MAX,
                    ..small(0)
                },
                CudaPlanInput {
                    largest_tensor_bytes: u64::MAX,
                    ..small(0)
                },
                CudaPlanInput {
                    max_shape: [1, 32769],
                    ..small(0)
                },
                CudaPlanInput {
                    max_shape: [u64::MAX, 16],
                    ..small(0)
                },
            ] {
                assert!(admit_cuda_plan(input, MemoryPolicy::default_share(), u64::MAX).is_err());
            }
            assert_eq!(held(), (0, 0));
            drop(reserve(0, 128).unwrap());
            assert_eq!(
                held(),
                (128, 0),
                "invalid metadata must not poison later planning"
            );
        }

        fn aggregate_overflows() {
            let large = 10_000_000_000_000_000_000;
            let first = reserve(large, u64::MAX).unwrap();
            assert!(matches!(
                reserve(large, u64::MAX),
                Err(EngError::MemoryPlan(_))
            ));
            assert_eq!(held(), (large + 128, 0));
            drop(first);
        }

        fn device_overflow() {
            let large = 10_000_000_000_000_000_000;
            let input = || CudaPlanInput {
                largest_tensor_bytes: large,
                ..small(0)
            };
            let first = admit_cuda_plan(input(), MemoryPolicy::default_share(), u64::MAX).unwrap();
            let peak = first.plan.device_peak;
            let first = first.into_cleanup(memory(u64::MAX), false).unwrap();
            let before = held();
            assert_eq!(before.1, peak);
            let second = admit_cuda_plan(input(), MemoryPolicy::default_share(), u64::MAX).unwrap();
            let host_after_issuance = held().0;
            assert!(matches!(
                second.into_cleanup(memory(u64::MAX), false),
                Err(EngError::MemoryPlan(_))
            ));
            assert_eq!(held(), (host_after_issuance, peak));
            drop(first);
            assert_eq!(held(), (host_after_issuance, peak));
        }

        fn transfer_and_peak_plan() {
            let limit = u64::try_from(PINNED_STAGE_LIMIT_BYTES).unwrap();
            let input = CudaPlanInput {
                host_base_bytes: 100,
                largest_tensor_bytes: limit + 4,
                packed_pair_bytes: 48,
                max_shape: [131072, 8192],
                device_weight_cache_bytes: 0,
            };
            let ticket = admit_cuda_plan(input, MemoryPolicy::default_share(), u64::MAX).unwrap();
            let (_, work) = CudaContext::fixed_projection_plan([131072, 8192]).unwrap();
            assert_eq!(ticket.plan.pinned_peak, 2 * limit);
            assert_eq!(ticket.plan.host_total, 100 + 2 * limit);
            assert_eq!(ticket.plan.device_peak, 2 * limit + work + limit + 4 + 48);
            let limits = ticket.transfer_limits();
            limits.check_shape([131072, 8192]).unwrap();
            assert!(limits.check_shape([131073, 8192]).is_err());
            assert!(limits.check_shape([1, 8193]).is_err());
            assert!(matches!(
                limits.check_shape([0, 1]),
                Err(EngError::BadRequest(_))
            ));
            limits
                .check_pageable(usize::try_from(limit + 4).unwrap())
                .unwrap();
            assert!(
                limits
                    .check_pageable(usize::try_from(limit + 5).unwrap())
                    .is_err()
            );
            limits.check_packed(32, 16).unwrap();
            assert!(limits.check_packed(32, 17).is_err());
            assert!(limits.check_packed(usize::MAX, 1).is_err());
        }

        fn stream_host_scratch_refunds_cache_at_exact_ceiling() {
            let input = CudaPlanInput {
                device_weight_cache_bytes: 772,
                ..small(100)
            };
            let mut ticket = admit_cuda_plan(input, MemoryPolicy::default_share(), 1_000).unwrap();
            let peak = ticket.plan.device_peak;
            assert_eq!(held(), (1_000, 0));
            let stream = super::super::ChunkStreamPlan {
                banks: crate::expert_tier::StreamBankPlan::new(256, 64, 128).unwrap(),
                max_pairs: 16,
                up_shape: [32, 32],
            };
            ticket.reserve_streaming(stream).unwrap();
            assert_eq!(ticket.device_weight_cache_bytes(), 132);
            assert_eq!(ticket.plan.host_total, 1_000);
            assert_eq!(ticket.plan.device_peak, peak - 64);
            assert_eq!(held(), (1_000, 0));
            ticket.into_cleanup(memory((peak - 64) * 4), false).unwrap()();
            assert_eq!(held(), (232, 0));
        }

        fn stream_real_shortage_keeps_reservation_unchanged() {
            let input = CudaPlanInput {
                device_weight_cache_bytes: 64,
                ..small(100)
            };
            let mut ticket = admit_cuda_plan(input, MemoryPolicy::default_share(), 867).unwrap();
            let stream = super::super::ChunkStreamPlan {
                banks: crate::expert_tier::StreamBankPlan::new(256, 64, 128).unwrap(),
                max_pairs: 16,
                up_shape: [32, 32],
            };
            let before = reservation_snapshot(&ticket);
            assert!(matches!(
                ticket.reserve_streaming(stream),
                Err(EngError::MemoryPlan(message)) if message.contains("host ceiling")
            ));
            assert_eq!(reservation_snapshot(&ticket), before);
            let peak = ticket.plan.device_peak;
            ticket.into_cleanup(memory(peak * 4), false).unwrap()();
            assert_eq!(held(), (164, 0));
        }

        fn stream_refund_uses_larger_of_host_and_device() {
            for (device_aux, host_scratch, cache, remaining_cache, host_total, extra_device) in [
                (128, 128, 4096, 3456, 4224, 0),
                (128, 64, 4096, 3456, 4160, 0),
                (64, 128, 64, 0, 768, 512),
                (64, 128, 0, 0, 768, 576),
            ] {
                let before = held();
                let input = CudaPlanInput {
                    device_weight_cache_bytes: cache,
                    ..small(0)
                };
                let mut ticket =
                    admit_cuda_plan(input, MemoryPolicy::default_share(), u64::MAX).unwrap();
                let peak = ticket.plan.device_peak;
                let stream = super::super::ChunkStreamPlan {
                    banks: crate::expert_tier::StreamBankPlan::new(256, device_aux, host_scratch)
                        .unwrap(),
                    max_pairs: 16,
                    up_shape: [32, 32],
                };
                ticket.reserve_streaming(stream).unwrap();
                assert_eq!(ticket.device_weight_cache_bytes(), remaining_cache);
                assert_eq!(ticket.plan.host_total, host_total);
                assert_eq!(ticket.plan.device_peak, peak + extra_device);
                assert_eq!(held(), (before.0 + host_total, before.1));
                let peak = ticket.plan.device_peak;
                ticket.into_cleanup(memory(peak * 4), true).unwrap()();
                assert_eq!(held(), (before.0 + remaining_cache, before.1));
            }
        }

        fn stream_refund_checks_overflow_before_mutation() {
            let mut ticket = reserve(0, u64::MAX).unwrap();
            let stream = super::super::ChunkStreamPlan {
                banks: crate::expert_tier::StreamBankPlan::new(u64::MAX / 2, 0, 0).unwrap(),
                max_pairs: 16,
                up_shape: [32, 32],
            };
            let before = reservation_snapshot(&ticket);
            assert!(matches!(
                ticket.reserve_streaming(stream),
                Err(EngError::MemoryPlan(message)) if message.contains("stream host reservation overflows")
            ));
            assert_eq!(reservation_snapshot(&ticket), before);

            let base = CudaPlan::new(
                CudaPlanInput {
                    packed_pair_bytes: 0,
                    ..small(0)
                },
                u64::MAX,
            )
            .unwrap()
            .device_peak;
            let input = CudaPlanInput {
                packed_pair_bytes: u64::MAX - base,
                ..small(0)
            };
            let mut ticket =
                admit_cuda_plan(input, MemoryPolicy::default_share(), u64::MAX).unwrap();
            assert_eq!(ticket.plan.device_peak, u64::MAX);
            let stream = super::super::ChunkStreamPlan {
                banks: crate::expert_tier::StreamBankPlan::new(256, 64, 128).unwrap(),
                ..stream
            };
            let before = reservation_snapshot(&ticket);
            assert!(matches!(
                ticket.reserve_streaming(stream),
                Err(EngError::MemoryPlan(message)) if message.contains("stream device reservation overflows")
            ));
            assert_eq!(reservation_snapshot(&ticket), before);
        }

        fn stream_repeat_reservation_is_rejected() {
            let input = CudaPlanInput {
                device_weight_cache_bytes: 1_000,
                ..small(0)
            };
            let mut ticket =
                admit_cuda_plan(input, MemoryPolicy::default_share(), u64::MAX).unwrap();
            let stream = super::super::ChunkStreamPlan {
                banks: crate::expert_tier::StreamBankPlan::new(256, 64, 128).unwrap(),
                max_pairs: 16,
                up_shape: [32, 32],
            };
            ticket.reserve_streaming(stream).unwrap();
            assert_eq!(ticket.device_weight_cache_bytes(), 360);
            let before = reservation_snapshot(&ticket);
            assert!(matches!(
                ticket.reserve_streaming(stream),
                Err(EngError::MemoryPlan(message)) if message.contains("already reserved")
            ));
            assert_eq!(reservation_snapshot(&ticket), before);
            let peak = ticket.plan.device_peak;
            ticket.into_cleanup(memory(peak * 4), false).unwrap()();
            assert_eq!(held(), (360, 0));
        }

        fn stream_aggregate_host_ceiling_is_preserved() {
            let first = reserve(100, 1_000).unwrap();
            let input = CudaPlanInput {
                device_weight_cache_bytes: 64,
                ..small(100)
            };
            let mut ticket = admit_cuda_plan(input, MemoryPolicy::default_share(), 1_000).unwrap();
            let stream = super::super::ChunkStreamPlan {
                banks: crate::expert_tier::StreamBankPlan::new(256, 64, 128).unwrap(),
                max_pairs: 16,
                up_shape: [32, 32],
            };
            let before = reservation_snapshot(&ticket);
            assert!(matches!(
                ticket.reserve_streaming(stream),
                Err(EngError::MemoryPlan(message)) if message.contains("host ceiling")
            ));
            assert_eq!(reservation_snapshot(&ticket), before);
            let peak = ticket.plan.device_peak;
            ticket.into_cleanup(memory(peak * 4), false).unwrap()();
            let peak = first.plan.device_peak;
            first.into_cleanup(memory(peak * 4), false).unwrap()();
            assert_eq!(held(), (264, 0));

            let other = reserve(10_000_000_000_000_000_000, u64::MAX).unwrap();
            let mut ticket = reserve(0, u64::MAX).unwrap();
            let stream = super::super::ChunkStreamPlan {
                banks: crate::expert_tier::StreamBankPlan::new(4_500_000_000_000_000_000, 0, 0)
                    .unwrap(),
                ..stream
            };
            let before = reservation_snapshot(&ticket);
            assert!(matches!(
                ticket.reserve_streaming(stream),
                Err(EngError::MemoryPlan(message)) if message.contains("aggregate stream host reservation overflows")
            ));
            assert_eq!(reservation_snapshot(&ticket), before);
            drop(other);
        }

        fn stream_reservation() {
            let stream = super::super::ChunkStreamPlan {
                banks: crate::expert_tier::StreamBankPlan::new(256, 64, 128).unwrap(),
                max_pairs: 16,
                up_shape: [32, 32],
            };
            let mut refused = reserve(100, 867).unwrap();
            assert!(matches!(
                refused.reserve_streaming(stream),
                Err(EngError::MemoryPlan(_))
            ));
            assert_eq!(held(), (228, 0), "stream refusal must be atomic");
            assert!(refused.stream_plan.is_none());
            drop(refused);

            let mut ticket = reserve(100, u64::MAX).unwrap();
            let old_device = ticket.plan.device_peak;
            ticket.reserve_streaming(stream).unwrap();
            assert_eq!(held(), (1096, 0));
            assert_eq!(ticket.plan.device_peak, old_device + 576);
            let before = held();
            assert!(ticket.reserve_streaming(stream).is_err());
            assert_eq!(held(), before, "a second reservation must not charge twice");
            let peak = ticket.plan.device_peak;
            let cleanup = ticket.into_cleanup(memory(peak * 4), true).unwrap();
            assert_eq!(held(), (1096 + peak, peak));
            cleanup();
            assert_eq!(
                held(),
                (328, 0),
                "native cleanup returns stream host credit"
            );

            let input = CudaPlanInput {
                device_weight_cache_bytes: 4096,
                ..small(0)
            };
            let mut ticket =
                admit_cuda_plan(input, MemoryPolicy::default_share(), u64::MAX).unwrap();
            let before = held();
            let peak = ticket.plan.device_peak;
            ticket.reserve_streaming(stream).unwrap();
            assert_eq!(ticket.device_weight_cache_bytes(), 4096 - 640);
            assert_eq!(
                ticket.plan.device_peak,
                peak - 64,
                "banks partition optional cache credit"
            );
            assert_eq!(held(), before);
            let cleanup = ticket.into_cleanup(memory(peak * 4), false).unwrap();
            cleanup();
            assert_eq!(held(), (328 + 4096 - 640, 0));
            let before_overflow = held();
            let mut overflow = reserve(0, u64::MAX).unwrap();
            let huge = super::super::ChunkStreamPlan {
                banks: crate::expert_tier::StreamBankPlan::new(u64::MAX / 2, 0, 0).unwrap(),
                ..stream
            };
            assert!(matches!(overflow.reserve_streaming(huge),
                Err(EngError::MemoryPlan(message)) if message.contains("stream host reservation overflows")));
            assert_eq!(held(), (before_overflow.0 + 128, 0));
            assert!(overflow.stream_plan.is_none());
            let peak = overflow.plan.device_peak;
            overflow.into_cleanup(memory(peak * 4), false).unwrap()();
            assert_eq!(held(), before_overflow);
        }

        fn poisoned_ledger() {
            let ticket = reserve(100, 1_000).unwrap();
            let peak = ticket.plan.device_peak;
            let cleanup = ticket.into_cleanup(memory(peak * 2), false).unwrap();
            let mut streaming = reserve(0, 1_000).unwrap();
            let before = held();
            let poisoned = std::thread::spawn(|| {
                let _held = HELD.lock().unwrap();
                panic!("intentional reservation-ledger poison");
            });
            assert!(poisoned.join().is_err());
            let snapshot = reservation_snapshot(&streaming);
            let stream = super::super::ChunkStreamPlan {
                banks: crate::expert_tier::StreamBankPlan::new(256, 64, 128).unwrap(),
                max_pairs: 16,
                up_shape: [32, 32],
            };
            assert!(matches!(
                streaming.reserve_streaming(stream),
                Err(EngError::MemoryPlan(message)) if message.contains("ledger is poisoned")
            ));
            assert_eq!(reservation_snapshot(&streaming), snapshot);
            cleanup();
            assert_eq!(held(), before);
            assert!(matches!(reserve(0, 1_000), Err(EngError::MemoryPlan(_))));
            assert_eq!(held(), before);
        }

        #[test]
        fn owned_reservation_contract() {
            let cases: [(&str, fn()); 16] = [
                ("host", host_boundaries),
                ("cleanup", cleanup_and_reload),
                ("discard", discarded_obligation),
                ("integrated", integrated_atomicity),
                ("concurrent", concurrent_same_snapshot),
                ("invalid", invalid_and_overflow),
                ("host-overflow", aggregate_overflows),
                ("device-overflow", device_overflow),
                ("limits", transfer_and_peak_plan),
                ("stream", stream_reservation),
                (
                    "stream_host_scratch_refunds_cache_at_exact_ceiling",
                    stream_host_scratch_refunds_cache_at_exact_ceiling,
                ),
                (
                    "stream_real_shortage_keeps_reservation_unchanged",
                    stream_real_shortage_keeps_reservation_unchanged,
                ),
                (
                    "stream_refund_uses_larger_of_host_and_device",
                    stream_refund_uses_larger_of_host_and_device,
                ),
                (
                    "stream_refund_checks_overflow_before_mutation",
                    stream_refund_checks_overflow_before_mutation,
                ),
                (
                    "stream_repeat_reservation_is_rejected",
                    stream_repeat_reservation_is_rejected,
                ),
                (
                    "stream_aggregate_host_ceiling_is_preserved",
                    stream_aggregate_host_ceiling_is_preserved,
                ),
            ];
            if let Ok(case) = std::env::var(CASE) {
                if case == "poison" {
                    poisoned_ledger();
                } else {
                    (cases.iter().find(|entry| entry.0 == case).unwrap().1)();
                }
                return;
            }
            // Fresh processes preserve the production ledger: no reset or forged capability.
            for case in cases.iter().map(|entry| entry.0).chain(["poison"]) {
                let output = Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", NAME, "--nocapture"])
                    .env(CASE, case)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "case={case} status={} stdout={} stderr={}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                println!("reservation case={case} passed");
            }
        }
    }
}
