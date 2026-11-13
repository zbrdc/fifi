//! The whole-model EXL3 load: every tensor of a stock pack that the resident step reads, on the
//! device, under the GGUF names of the inventory.
//!
//! One layer at a time, a mixer (full attention or DeltaNet) and its MoE block; then the embedding
//! table, the output norm and the head. A pack that does not fit the family is refused before the
//! first upload where the headers allow it (config, expert tensors present); every matrix is
//! checked against the geometry before its own upload. The result is an ordinary
//! [`CudaWeights`] of the `Exl3` layout, so the GGUF loads and their readers are untouched.

use std::path::Path;

use super::{
    super::{
        exl3_pack::{EMBED_NAME, HEAD_PREFIX, NORM_NAME, Root, inspect_pack, inspect_pack_split},
        mtp_pack::inspect_mtp_pack,
    },
    dims::StepDims,
    exl3_bank::ExpertStore,
    exl3_moe_source::{check_moe_source_geometry, moe_weights, moe_weights_banked},
    exl3_source::{PackReader, attention_weights, delta_weights},
    layer_table::LayerTable,
    mixer_geometry::MixerGeometry,
    moe::MoeDims,
    mtp_load::mtp_weights,
    pinned::ROTARY_DIM,
    resources::{GlobalNames, GlobalSpecs, STEP_ROWS},
};
use crate::{
    EngError,
    capability::probe::probe_host,
    family::{
        AttentionKind, FamilySpec, QWEN3_6_35B_A3B,
        gguf_model::Inventory,
        project::cuda::{CudaProjector, CudaWeight, CudaWeights, Reserve},
        qwen36::{host::HostShape, settings::ExpertBankSettings},
    },
    weights::exl3::Checkpoint,
};

pub(super) struct ModelFacts<'a> {
    pub(super) spec: &'a FamilySpec,
    pub(super) dims: &'a StepDims,
    pub(super) inventory: &'a Inventory,
}

pub(super) fn load_exl3_dir(
    projector: &CudaProjector,
    dir: &Path,
    reserve: Reserve,
) -> Result<CudaWeights, EngError> {
    load_exl3_dir_with_head(projector, dir, None, reserve)
}

/// The banked load's result: the dense weights plus `slots`-member banks as ordinary
/// [`CudaWeights`], every routed expert's bank-layout bytes in the pageable host `store`, and
/// the walk figures the start line echoes.
pub(super) struct BankedModel {
    pub(super) weights: CudaWeights,
    pub(super) store: ExpertStore,
    pub(super) slots: usize,
    /// The model's routed experts per layer (the store holds every one; the banks hold
    /// `slots`).
    pub(super) experts: usize,
    pub(super) routed_bytes: u64,
    pub(super) dense_bytes: u64,
}

/// The banked EXL3 load: the dense figure is admitted up front, and each routed bank admits
/// its own slot bytes before its allocation (the start-time arithmetic already sized the
/// slots against the free memory; the per-bank admits are the race safety net).
pub(super) fn load_exl3_banked_dir(
    projector: &CudaProjector,
    dir: &Path,
    slots: usize,
    reserve: Reserve,
) -> Result<BankedModel, EngError> {
    let checkpoint = Checkpoint::open(dir)?;
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    let dims = StepDims::new(&shape, &inventory)?;
    let facts = ModelFacts {
        spec: &QWEN3_6_35B_A3B,
        dims: &dims,
        inventory: &inventory,
    };
    let reader = PackReader::new(projector, &checkpoint, reserve);
    let report = inspect_pack_split(&checkpoint.source, &shape)?;
    let dense = report.device_bytes.saturating_sub(report.expert_bytes);
    reader.admit(dense)?;
    let layers = LayerTable::new(&dims, &inventory)?.len();
    let mut store = ExpertStore::with_layers(layers);
    let weights = model_weights_admitted(&reader, &facts, Some((slots, &mut store)))?;
    // The store never changes again: page-lock it so the bank refills DMA without staging.
    // A failed pin is best-effort: refills then copy through pageable staging, and the
    // operator sees it here rather than inferring it from refill timings.
    if !store.pin_all() {
        eprintln!("q36 cuda: EXL3 host store pin failed; bank refills will stage pageable copies");
    }
    Ok(BankedModel {
        weights: CudaWeights::from_exl3(weights)?,
        store,
        slots,
        experts: shape.moe.experts,
        routed_bytes: report.expert_bytes,
        dense_bytes: dense,
    })
}

/// The outcome of a possibly-banked EXL3 load: the resident load as before, or the banked
/// load's model (whose decode wiring is the next slice — the start refuses to serve it).
pub(super) enum DecidedLoad {
    Resident(CudaWeights),
    Banked(Box<BankedModel>),
}

/// [`load_exl3_dir`] under the `expert_bank` policy: with the member absent (or `auto` and
/// the pack fits) this is the resident load; otherwise the header walk's figures, the free
/// device memory and the host RAM figures decide the slot count and the banked load runs.
pub(super) fn load_exl3_dir_decided(
    projector: &CudaProjector,
    dir: &Path,
    expert_bank: Option<&ExpertBankSettings>,
    reserve: Reserve,
) -> Result<DecidedLoad, EngError> {
    let Some(bank) = expert_bank else {
        return Ok(DecidedLoad::Resident(load_exl3_dir(
            projector, dir, reserve,
        )?));
    };
    let checkpoint = Checkpoint::open(dir)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    let report = inspect_pack_split(&checkpoint.source, &shape)?;
    let host_facts = probe_host();
    let host = (
        host_facts.mem_total_bytes.clone(),
        host_facts.mem_available_bytes.clone(),
    );
    let host = match host {
        (
            crate::capability::Probe::Measured(total),
            crate::capability::Probe::Measured(available),
        ) => Some((total, available)),
        _ => None,
    };
    let free = u64::try_from(projector.memory_info()?.free).unwrap_or(u64::MAX);
    let dense = report.device_bytes.saturating_sub(report.expert_bytes);
    let layers = shape.kinds.len();
    let experts = shape.moe.experts;
    let decision = super::exl3_bank::decide(
        bank.slots,
        bank.mode,
        dense,
        report.expert_bytes,
        layers,
        experts,
        free,
        reserve,
        host,
    )?;
    // A bank smaller than one token's picks cannot hold them together: the resolver would
    // refuse every decode token and every prefill sub-block after a successful boot. Refuse
    // at load, where the operator still sees the memory figure that produced the count.
    if let Some(admission) = &decision
        && admission.slots < shape.moe.top_k
    {
        return Err(EngError::BadRequest(format!(
            "expert_bank: the fit admitted {} slots but one token picks {} experts; \
             free device memory or set expert_bank.slots to at least {}",
            admission.slots, shape.moe.top_k, shape.moe.top_k
        )));
    }
    match decision {
        None => Ok(DecidedLoad::Resident(load_exl3_dir(
            projector, dir, reserve,
        )?)),
        Some(admission) => Ok(DecidedLoad::Banked(Box::new(load_exl3_banked_dir(
            projector,
            dir,
            admission.slots,
            reserve,
        )?))),
    }
}

/// Like [`load_exl3_dir`], and with `head_dir` set also the MTP head found in that pack, beside
/// the model's weights. The head's pack may differ from the model's in rate and codebook: every
/// matrix carries its own.
pub(super) fn load_exl3_dir_with_head(
    projector: &CudaProjector,
    dir: &Path,
    head_dir: Option<&Path>,
    reserve: Reserve,
) -> Result<CudaWeights, EngError> {
    let checkpoint = Checkpoint::open(dir)?;
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    let dims = StepDims::new(&shape, &inventory)?;
    let facts = ModelFacts {
        spec: &QWEN3_6_35B_A3B,
        dims: &dims,
        inventory: &inventory,
    };
    let reader = PackReader::new(projector, &checkpoint, reserve);
    let mut weights = model_weights(&reader, &facts)?;
    if let Some(head_dir) = head_dir {
        let head = Checkpoint::open(head_dir)?;
        let reader = PackReader::new(projector, &head, reserve).under(Root::Mtp);
        weights.extend(head_weights(&reader, &facts)?);
    }
    CudaWeights::from_exl3(weights)
}

pub(super) fn load_exl3_model(
    reader: &PackReader<'_>,
    facts: &ModelFacts<'_>,
) -> Result<CudaWeights, EngError> {
    CudaWeights::from_exl3(model_weights(reader, facts)?)
}

/// The MTP head of the pack behind `reader` (which reads the MTP root), under the names of a
/// block numbered after the model's own layers.
fn head_weights(
    reader: &PackReader<'_>,
    facts: &ModelFacts<'_>,
) -> Result<Vec<CudaWeight>, EngError> {
    let config = &reader.checkpoint.model_config;
    facts.spec.check_architecture_config(config)?;
    let layers = mtp_layers(config)?;
    let report = inspect_mtp_pack(&reader.checkpoint.source, &facts.dims.shape, layers)?;
    reader.admit(report.device_bytes)?;
    let moe = MoeDims::new(facts.dims, STEP_ROWS);
    check_moe_source_geometry(config, &moe)?;
    let table = LayerTable::new(facts.dims, facts.inventory)?;
    let first_full = (0..table.len())
        .find(|&layer| matches!(table.slot(layer).map(|s| s.kind), Ok(AttentionKind::Full)))
        .ok_or_else(|| {
            EngError::Checkpoint("EXL3 load: the model has no full-attention layer".into())
        })?;
    let block = table.single(first_full, table.len())?;
    let geometry = MixerGeometry::new(facts.dims)?;
    mtp_weights(reader, &block, &geometry, &moe)
}

/// `mtp_num_hidden_layers` of the config, which this load requires to be 1.
fn mtp_layers(config: &[u8]) -> Result<usize, EngError> {
    let value: serde_json::Value = serde_json::from_slice(config)
        .map_err(|error| EngError::Checkpoint(format!("EXL3 MTP config: {error}")))?;
    let layers = value
        .get("text_config")
        .and_then(|text| text.get("mtp_num_hidden_layers"))
        .and_then(serde_json::Value::as_u64);
    match layers {
        Some(1) => Ok(1),
        other => Err(EngError::Checkpoint(format!(
            "EXL3 MTP config: mtp_num_hidden_layers is {other:?}, the load needs 1"
        ))),
    }
}

/// Every tensor of the model's own pack, on the device: the config and geometry checks, the
/// header-only walk, the whole-figure admission, then the per-layer loads.
fn model_weights(
    reader: &PackReader<'_>,
    facts: &ModelFacts<'_>,
) -> Result<Vec<CudaWeight>, EngError> {
    // Every tensor is checked against the model from the headers alone, and the whole load is
    // admitted against the free memory, before the first upload.
    let report = inspect_pack(&reader.checkpoint.source, &facts.dims.shape)?;
    reader.admit(report.device_bytes)?;
    model_weights_admitted(reader, facts, None)
}

/// The per-layer loads after admission (the resident arm admitted the whole pack figure, the
/// banked arm the dense figure plus its per-bank admits): the config and geometry checks, one
/// mixer and one MoE block per layer (resident, or banked with `slots` members per bank and
/// the host store fill), then the globals.
fn model_weights_admitted(
    reader: &PackReader<'_>,
    facts: &ModelFacts<'_>,
    mut banked: Option<(usize, &mut ExpertStore)>,
) -> Result<Vec<CudaWeight>, EngError> {
    let model_config = &reader.checkpoint.model_config;
    if model_config.len() as u64 > crate::weights::exl3::CONFIG_LIMIT {
        return Err(EngError::MemoryPlan(
            "EXL3 model config exceeds 1 MiB".into(),
        ));
    }
    facts.spec.check_architecture_config(model_config)?;
    let moe = MoeDims::new(facts.dims, STEP_ROWS);
    check_moe_source_geometry(model_config, &moe)?;
    let table = LayerTable::new(facts.dims, facts.inventory)?;
    let geometry = MixerGeometry::new(facts.dims)?;
    let mut weights = Vec::new();
    for layer in 0..table.len() {
        let mixer = match table.slot(layer)?.kind {
            AttentionKind::Full => attention_weights(reader, &table, layer, &geometry)?,
            AttentionKind::DeltaNet => delta_weights(reader, &table, layer, &geometry)?,
            other => {
                return Err(EngError::Checkpoint(format!(
                    "EXL3 load: layer {layer} has attention kind {other:?}, which has no mixer"
                )));
            }
        };
        weights.extend(mixer);
        match banked.as_mut() {
            None => weights.extend(moe_weights(reader, &table, layer, &moe)?),
            Some((slots, store)) => weights.extend(moe_weights_banked(
                reader, &table, layer, &moe, *slots, store,
            )?),
        }
    }
    weights.extend(global_weights(reader, facts)?);
    Ok(weights)
}

/// The embedding table is kept in its stored BF16.
fn global_weights(
    reader: &PackReader<'_>,
    facts: &ModelFacts<'_>,
) -> Result<Vec<CudaWeight>, EngError> {
    let names = GlobalNames::new(facts.inventory)?;
    let specs = GlobalSpecs::new(facts.dims);
    let embed_shape = [specs.embed.rows as u32, specs.embed.k as u32];
    Ok(vec![
        reader.table(names.embed, EMBED_NAME, &embed_shape)?,
        reader.norm(names.norm, NORM_NAME, specs.norm_values)?,
        reader.exl3(names.head, HEAD_PREFIX, &specs.head, None)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::mtp_layers;
    use crate::EngError;

    fn layers(config: &str) -> Result<usize, EngError> {
        mtp_layers(config.as_bytes())
    }

    #[test]
    fn the_head_needs_exactly_one_block_in_the_config() {
        assert_eq!(
            layers(r#"{"text_config":{"mtp_num_hidden_layers":1}}"#).unwrap(),
            1
        );
        for config in [
            r#"{"text_config":{"mtp_num_hidden_layers":0}}"#,
            r#"{"text_config":{"mtp_num_hidden_layers":2}}"#,
            r#"{"text_config":{"mtp_num_hidden_layers":"1"}}"#,
            r#"{"text_config":{"mtp_num_hidden_layers":-1}}"#,
            r#"{"text_config":{"mtp_num_hidden_layers":1.5}}"#,
            r#"{"text_config":{}}"#,
            r#"{"mtp_num_hidden_layers":1}"#,
            r#"[]"#,
            r#"not json"#,
            "",
        ] {
            assert!(
                matches!(layers(config), Err(EngError::Checkpoint(_))),
                "{config}"
            );
        }
    }
}
