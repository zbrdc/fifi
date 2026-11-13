//! Source names, geometry checks and device upload of one Qwen3.6 MoE block in a stock EXL3
//! pack.
//!
//! Stock packs keep the mixed-precision tensors under
//! `model.language_model.layers.<layer>.mlp...` with the router and the shared-expert gate
//! stored as plain tensors. This module owns the name map, the config-vs-dims refusal for the
//! block and the upload: the router and the gate as F32 vectors, the routed experts of each
//! projection as one bank, the shared expert as three single matrices.

use crate::family::{
    exl3_pack::{ExpertPart, expert_name_in, router_name_in, shared_gate_name_in, shared_name_in},
    qwen36::cuda::{
        exl3_bank::{ExpertStore, PartStore},
        exl3_source::PackReader,
        layer_table::LayerTable,
        mixer_geometry::DenseSpec,
        moe::MoeDims,
    },
};
use crate::{
    EngError,
    family::{
        gguf_model::Role,
        project::cuda::{CudaWeight, exl3::exl3_member_staging},
    },
    weights::exl3::Matrix,
};

/// Output rows and input length each expert projection must have.
pub(super) struct ExpertShape {
    pub(super) rows: usize,
    pub(super) k: usize,
}

pub(super) fn expert_shapes(moe: &MoeDims) -> ExpertPlan {
    let (hidden, inter, shared_inter) = (moe.hidden, moe.inter, moe.shared_inter);
    ExpertPlan {
        gate: ExpertShape {
            rows: inter,
            k: hidden,
        },
        up: ExpertShape {
            rows: inter,
            k: hidden,
        },
        down: ExpertShape {
            rows: hidden,
            k: inter,
        },
        shared_gate: ExpertShape {
            rows: shared_inter,
            k: hidden,
        },
        shared_up: ExpertShape {
            rows: shared_inter,
            k: hidden,
        },
        shared_down: ExpertShape {
            rows: hidden,
            k: shared_inter,
        },
        router_rows: moe.n_expert,
        router_k: hidden,
    }
}

pub(super) struct ExpertPlan {
    pub(super) gate: ExpertShape,
    pub(super) up: ExpertShape,
    pub(super) down: ExpertShape,
    pub(super) shared_gate: ExpertShape,
    pub(super) shared_up: ExpertShape,
    pub(super) shared_down: ExpertShape,
    pub(super) router_rows: usize,
    pub(super) router_k: usize,
}

/// `model_config` is the checkpoint's config.json bytes; the MoE fields live
/// under `text_config`, matching the attention source's contract.
pub(in crate::family) fn check_moe_source_geometry(
    model_config: &[u8],
    moe: &MoeDims,
) -> Result<(), EngError> {
    let refuse = |field: &str, have: usize, want: usize| {
        Err(EngError::Checkpoint(format!(
            "EXL3 MoE source: config {field} is {have}, the model needs {want}"
        )))
    };
    let config: serde_json::Value = serde_json::from_slice(model_config).map_err(|error| {
        EngError::Checkpoint(format!("EXL3 MoE source: config is not JSON: {error}"))
    })?;
    let text = config
        .get("text_config")
        .ok_or_else(|| EngError::Checkpoint("EXL3 MoE source: config has no text_config".into()))?;
    let field = |name: &str| -> Result<usize, EngError> {
        text.get(name)
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                EngError::Checkpoint(format!(
                    "EXL3 MoE source: config text_config.{name} is missing or not a count"
                ))
            })
    };
    let experts = field("num_experts")?;
    if experts != moe.n_expert {
        return refuse("num_experts", experts, moe.n_expert);
    }
    let inter = field("moe_intermediate_size")?;
    if inter != moe.inter {
        return refuse("moe_intermediate_size", inter, moe.inter);
    }
    let shared_inter = field("shared_expert_intermediate_size")?;
    if shared_inter != moe.shared_inter {
        return refuse(
            "shared_expert_intermediate_size",
            shared_inter,
            moe.shared_inter,
        );
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn stored_shape(rows: usize, k: usize) -> (usize, usize) {
    (rows, k)
}

pub(in crate::family) fn moe_weights(
    reader: &PackReader<'_>,
    table: &LayerTable,
    layer: usize,
    moe: &MoeDims,
) -> Result<Vec<CudaWeight>, EngError> {
    moe_weights_inner(reader, table, layer, moe, None)
}

/// The banked arm: `slots` members per routed bank on the device, and every expert of every
/// projection staged into the host `store`'s entry for this layer (gate, up, down order), in
/// the device-bank byte layout a later miss copy moves unchanged.
pub(in crate::family) fn moe_weights_banked(
    reader: &PackReader<'_>,
    table: &LayerTable,
    layer: usize,
    moe: &MoeDims,
    slots: usize,
    store: &mut ExpertStore,
) -> Result<Vec<CudaWeight>, EngError> {
    moe_weights_inner(reader, table, layer, moe, Some((slots, store)))
}

fn moe_weights_inner(
    reader: &PackReader<'_>,
    table: &LayerTable,
    layer: usize,
    moe: &MoeDims,
    banked: Option<(usize, &mut ExpertStore)>,
) -> Result<Vec<CudaWeight>, EngError> {
    let plan = expert_shapes(moe);
    let root = reader.root();
    let name = |role| table.name(layer, role).map(str::to_owned);
    let mut weights = Vec::with_capacity(8);
    let router = [plan.router_rows as u32, plan.router_k as u32];
    weights.push(reader.vector(
        name(Role::FfnGateInp)?,
        &router_name_in(root, layer),
        &router,
        Ok,
    )?);
    let gate_shape = [1, moe.hidden as u32];
    weights.push(reader.vector(
        name(Role::FfnGateInpShexp)?,
        &shared_gate_name_in(root, layer),
        &gate_shape,
        Ok,
    )?);
    let routed_count = banked.as_ref().map_or(moe.n_expert, |(slots, _)| *slots);
    let mut parts = [None, None, None];
    for (index, (role, part, shape)) in [
        (Role::FfnGateExps, ExpertPart::Gate, &plan.gate),
        (Role::FfnUpExps, ExpertPart::Up, &plan.up),
        (Role::FfnDownExps, ExpertPart::Down, &plan.down),
    ]
    .into_iter()
    .enumerate()
    {
        let matrix = |expert: usize| {
            let matrix = reader.matrix(&expert_name_in(root, layer, expert, part))?;
            if matrix.shape() != [shape.k, shape.rows] {
                return Err(EngError::Checkpoint(format!(
                    "EXL3 MoE source: expert {expert} of layer {layer} is {:?}, expected {:?}",
                    matrix.shape(),
                    [shape.k, shape.rows]
                )));
            }
            Ok(matrix)
        };
        weights.push(reader.bank(name(role)?, routed_count, matrix)?);
        if banked.is_some() {
            parts[index] = Some(fill_part_store(layer, moe, matrix)?);
        }
    }
    if let Some((_, store)) = banked {
        store.push_layer(parts.map(|part| part.expect("the loop fills every part")));
    }
    for (role, part, shape) in [
        (Role::FfnGateShexp, ExpertPart::Gate, &plan.shared_gate),
        (Role::FfnUpShexp, ExpertPart::Up, &plan.shared_up),
        (Role::FfnDownShexp, ExpertPart::Down, &plan.shared_down),
    ] {
        let want = DenseSpec {
            role,
            rows: shape.rows,
            k: shape.k,
        };
        let prefix = shared_name_in(root, layer, part);
        weights.push(reader.exl3(name(role)?, &prefix, &want, None)?);
    }
    Ok(weights)
}

/// Stages every expert of one projection into a fresh part store. The first matrix fixes the
/// store's layout; every later member must match it in shape, rate and codebook (the bank
/// upload checks the same for the members it uploads, this covers the rest).
fn fill_part_store(
    layer: usize,
    moe: &MoeDims,
    matrix: impl Fn(usize) -> Result<Matrix, EngError>,
) -> Result<PartStore, EngError> {
    let first = matrix(0)?;
    let (shape, bits_twice, book) = (first.shape(), first.bitrate_twice(), first.codebook());
    let mut store = PartStore::new(shape, bits_twice, moe.n_expert);
    let staged = exl3_member_staging(&first)?;
    store.put(0, &staged)?;
    for expert in 1..moe.n_expert {
        let member = matrix(expert)?;
        if member.shape() != shape
            || member.bitrate_twice() != bits_twice
            || member.codebook() != book
            || member.bias().is_some()
        {
            return Err(EngError::Checkpoint(format!(
                "EXL3 expert bank: expert {expert} of layer {layer} differs from expert 0 in \
                 shape, rate, codebook or bias"
            )));
        }
        store.put(expert, &exl3_member_staging(&member)?)?;
    }
    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen36::exl3_pack::{expert_name, router_name, shared_name};

    fn moe() -> MoeDims {
        MoeDims {
            rows: 1,
            hidden: 1024,
            n_expert: 256,
            top_k: 10,
            inter: 512,
            shared_inter: 64,
        }
    }

    fn config(experts: u64, inter: u64, shared_inter: u64) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({"text_config": {
            "num_experts": experts,
            "moe_intermediate_size": inter,
            "shared_expert_intermediate_size": shared_inter
        }}))
        .unwrap()
    }

    #[test]
    fn names_follow_the_stock_pack_layout() {
        assert_eq!(
            router_name(7),
            "model.language_model.layers.7.mlp.gate.weight"
        );
        assert_eq!(
            expert_name(7, 123, ExpertPart::Gate),
            "model.language_model.layers.7.mlp.experts.123.gate_proj"
        );
        assert_eq!(
            expert_name(0, 255, ExpertPart::Down),
            "model.language_model.layers.0.mlp.experts.255.down_proj"
        );
        assert_eq!(
            shared_name(39, ExpertPart::Up),
            "model.language_model.layers.39.mlp.shared_expert.up_proj"
        );
    }

    #[test]
    fn expert_shapes_read_hidden_and_write_intermediate() {
        let plan = expert_shapes(&moe());
        assert_eq!(stored_shape(plan.gate.rows, plan.gate.k), (512, 1024));
        assert_eq!(stored_shape(plan.up.rows, plan.up.k), (512, 1024));
        assert_eq!(stored_shape(plan.down.rows, plan.down.k), (1024, 512));
        assert_eq!(
            stored_shape(plan.shared_gate.rows, plan.shared_gate.k),
            (64, 1024)
        );
        assert_eq!(
            stored_shape(plan.shared_up.rows, plan.shared_up.k),
            (64, 1024)
        );
        assert_eq!(
            stored_shape(plan.shared_down.rows, plan.shared_down.k),
            (1024, 64)
        );
        assert_eq!((plan.router_rows, plan.router_k), (256, 1024));
    }

    #[test]
    fn matching_config_passes_and_each_field_refuses() {
        check_moe_source_geometry(&config(256, 512, 64), &moe()).unwrap();
        for (experts, inter, shared) in [(8, 512, 64), (256, 8, 64), (256, 512, 8)] {
            assert!(matches!(
                check_moe_source_geometry(&config(experts, inter, shared), &moe()),
                Err(EngError::Checkpoint(_))
            ));
        }
    }

    #[test]
    fn missing_or_non_numeric_fields_refuse() {
        assert!(matches!(
            check_moe_source_geometry(b"{}", &moe()),
            Err(EngError::Checkpoint(_))
        ));
        // From here every fixture carries text_config, so the refusals below
        // exercise the numeric-field path itself, not the missing-section gate.
        let text = |body: serde_json::Value| {
            serde_json::to_vec(&serde_json::json!({"text_config": body})).unwrap()
        };
        let non_numeric = text(serde_json::json!({
            "num_experts": "256",
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 64
        }));
        assert!(matches!(
            check_moe_source_geometry(&non_numeric, &moe()),
            Err(EngError::Checkpoint(_))
        ));
        let one_field_dropped = text(serde_json::json!({
            "num_experts": 256,
            "shared_expert_intermediate_size": 64
        }));
        assert!(matches!(
            check_moe_source_geometry(&one_field_dropped, &moe()),
            Err(EngError::Checkpoint(_))
        ));
    }
}
