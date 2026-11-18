//! The per-card weight-upload manifest: the load plan turned into exact
//! device bytes and slice plans. This is the host side of the real
//! GGUF-to-card loading — the loader drives `load` for whole tensors and
//! `load_sliced` for the planned slices, consuming [`TensorSlice::segments`]
//! and nothing else, so the file bytes it uploads are exactly the bytes
//! these plans describe.
//!
//! The manifest never decides placement: it walks [`shard_loads`], which
//! derives every range from the SAME [`ShardTable`] the planner selected
//! and the executor consumes. Byte counts are the planner's exact
//! [`plan_slice_rows`] figures, not the proportional estimate admission
//! uses while comparing candidates, so a receipt built from these totals
//! can be diffed against CUDA's observed allocations without estimation
//! slack. The DeltaNet `attn_qkv` and `conv1d` slices over-cover the file
//! by one replicated key copy per card (the kernel pairs value head `h`
//! with key head `h % 16`); their `sliced_bytes` deliberately count the
//! replicated copy, because that is what the card uploads and holds.

use crate::EngError;
use crate::family::gguf_model::{Inventory, Role};
use crate::family::qwen36::cuda::shard::{ShardLoad, ShardTable, shard_loads};
use crate::kernels::cuda::plan::{TensorSlice, native_device_bytes, plan_slice_rows};
use crate::weights::{GgufHeader, GgufType};
use std::collections::HashMap;

/// The native size of a full tensor from the checkpoint's own quantization.
pub(super) struct TensorWeight {
    pub(super) ty: GgufType,
    pub(super) bytes: u64,
    pub(super) cols: u64,
    pub(super) rows: u64,
}

/// Freeze the per-tensor sizes once, before comparing any shard tables.
pub(super) struct ModelWeights<'a> {
    pub(super) inventory: &'a Inventory,
    tensors: HashMap<&'a str, TensorWeight>,
    pub(super) column_block: usize,
}

impl<'a> ModelWeights<'a> {
    pub(super) fn from_checkpoint(
        inventory: &'a Inventory,
        header: &GgufHeader,
    ) -> Result<Self, EngError> {
        Self::from_types(inventory, |name, _| {
            header
                .tensor(name)
                .map(|t| t.ggml_type())
                .ok_or_else(|| EngError::Checkpoint(format!("weight plan: missing tensor {name}")))
        })
    }

    pub(super) fn from_types(
        inventory: &'a Inventory,
        ty_for: impl Fn(&str, &[GgufType]) -> Result<GgufType, EngError>,
    ) -> Result<Self, EngError> {
        let mut tensors = HashMap::with_capacity(inventory.tensors().len());
        let mut column_block = 1;
        for tensor in inventory.tensors() {
            let ty = ty_for(&tensor.name, tensor.allowed_types())?;
            // Only cuts along a matrix's K dimension must align to its native block.
            // The supported GGUF block widths are powers of two.
            if matches!(
                tensor.role,
                Role::AttnOutput
                    | Role::SsmOut
                    | Role::FfnDownExps
                    | Role::FfnDownShexp
                    | Role::SsmA
                    | Role::SsmDtBias
            ) {
                column_block = column_block.max(ty.block_elems() as usize);
            }
            let bytes = native_device_bytes(&tensor.name, ty, &tensor.dims)?;
            tensors.insert(
                tensor.name.as_str(),
                TensorWeight {
                    ty,
                    bytes,
                    cols: tensor.dims[0],
                    rows: tensor.dims.get(1).copied().unwrap_or(1),
                },
            );
        }
        Ok(Self {
            inventory,
            tensors,
            column_block,
        })
    }

    #[cfg(test)]
    pub(super) fn uniform(inventory: &'a Inventory, quant: GgufType) -> Result<Self, EngError> {
        Self::from_types(inventory, |_, allowed| {
            Ok(if allowed.contains(&quant) {
                quant
            } else {
                allowed[0]
            })
        })
    }

    pub(super) fn get(&self, name: &str) -> Result<&TensorWeight, EngError> {
        self.tensors
            .get(name)
            .ok_or_else(|| EngError::Checkpoint(format!("weight plan: missing tensor {name}")))
    }
}

/// One inventory tensor as one card loads it.
#[derive(Debug)]
pub(crate) struct CardEntry {
    pub(crate) name: String,
    /// The tensor's GGUF type.
    pub(crate) ty: GgufType,
    /// GGUF dims of the FULL tensor, GGUF order.
    pub(crate) dims: Vec<u64>,
    pub(crate) load: ShardLoad,
    /// Exact device-resident bytes: the whole tensor's, or the planned
    /// slice's. Whole tensors count [`native_device_bytes`], the same
    /// function the slice planner builds on, so the two can never
    /// disagree.
    pub(crate) device_bytes: u64,
    /// The planned slice when the card loads a slice.
    pub(crate) slice: Option<TensorSlice>,
}

/// The upload manifest of one card over the whole inventory: every tensor
/// the model holds, in inventory order.
#[derive(Debug)]
pub(crate) struct CardLoad {
    pub(crate) entries: Vec<CardEntry>,
    /// Exact bytes of this card's sliced tensors.
    pub(crate) sliced_bytes: u64,
    /// Exact bytes replicated on every card (norms, router, per-head
    /// vectors, shared-expert gate scalars).
    pub(crate) whole_bytes: u64,
    /// Exact bytes only card 0 uploads (the embedding table); 0 on every
    /// other card.
    pub(crate) solo_bytes: u64,
}

impl CardLoad {
    /// The bytes this card uploads in total.
    pub(crate) fn upload_bytes(&self) -> u64 {
        self.sliced_bytes + self.whole_bytes + self.solo_bytes
    }
}

/// Build a manifest for fixtures with a uniform matrix quantization.
#[cfg(test)]
pub(crate) fn card_loads(
    table: &ShardTable,
    inventory: &Inventory,
    quant: GgufType,
) -> Result<Vec<CardLoad>, EngError> {
    checkpoint_card_loads(table, &ModelWeights::uniform(inventory, quant)?)
}

/// The upload manifest uses the same checkpoint metadata as placement.
pub(super) fn checkpoint_card_loads(
    table: &ShardTable,
    weights: &ModelWeights,
) -> Result<Vec<CardLoad>, EngError> {
    (0..table.cards())
        .map(|card| card_load(table, weights, card))
        .collect()
}

fn card_load(
    table: &ShardTable,
    weights: &ModelWeights,
    card: usize,
) -> Result<CardLoad, EngError> {
    let inventory = weights.inventory;
    let mut sliced = 0u64;
    let mut whole = 0u64;
    let mut solo = 0u64;
    let mut entries = Vec::with_capacity(inventory.tensors().len());
    for (name, load) in shard_loads(table, inventory, card)? {
        let tensor = inventory
            .get(&name)
            .ok_or_else(|| EngError::Checkpoint(format!("the load plan lost tensor {name}")))?;
        let weight = weights.get(&name)?;
        let ty = weight.ty;
        let (device_bytes, slice) = match &load {
            ShardLoad::Whole | ShardLoad::CardZeroOnly => (weight.bytes, None),
            ShardLoad::Sliced(spec) => {
                let plan = plan_slice_rows(&name, ty, &tensor.dims, spec.cols.clone(), &spec.rows)?;
                (plan.bytes, Some(plan))
            }
        };
        match load {
            ShardLoad::Whole => whole += device_bytes,
            // The entry stays on every card's manifest (the load plan is
            // card-shaped), but only card 0 counts and uploads it.
            ShardLoad::CardZeroOnly => solo += u64::from(card == 0) * device_bytes,
            ShardLoad::Sliced(_) => sliced += device_bytes,
        }
        entries.push(CardEntry {
            name,
            ty,
            dims: tensor.dims.clone(),
            load,
            device_bytes,
            slice,
        });
    }
    Ok(CardLoad {
        entries,
        sliced_bytes: sliced,
        whole_bytes: whole,
        solo_bytes: solo,
    })
}
