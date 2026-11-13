//! The CUDA trunk loader: the tensors of the Flash-Next EXL3 pack that stay on the device for the
//! whole run, uploaded through the Qwen3.6 EXL3 source reader.
//!
//! WHAT: the trunk is everything the header walk visits except the routed experts (the slot bank
//! owns them), the embedding and the tensors a host feed reads (the PLE norms and convolution).
//! [`TrunkList`] is that set, taken from the recording walk (`inspect_flash_pack_listing`), each
//! tensor with the [`Form`] the device holds it in. The forms follow the walk's own device-byte
//! rules, so the list's bytes are the plan's trunk bytes by construction and [`TrunkList::new`]
//! checks the sums against the report. A walk tensor that is not in the role tables of this file is
//! a refusal naming it, never a default form. The upload (native-cuda builds) goes through
//! `PackReader`, the reader the Qwen3.6 loader uses: `matrix` plus `upload_exl3` for every EXL3
//! matrix (`lm_head`, the DeltaNet and QSA projections, the shared expert), `table` for the weights
//! the kernels take as stored (hyper-connection sites, the two PLE projections), `vector` for F32
//! widening (router, shared-expert gate, `in_proj_a`/`in_proj_b`, DeltaNet vectors, the indexer
//! norms), `norm` for the `1 + w` norms of `attn_prep`. `A_log` is stored as `-exp(A_log)`
//! (`decay_from_log`), the form the scan kernel reads. DeltaNet matrices keep the pack's grouped
//! head order: the Flash-Next scan reads it directly (`GdnKeyHeads::Grouped`), unlike the
//! Qwen3.6 reorder. Every weight is keyed by its pack name in the returned `CudaWeights`; a matrix
//! by its prefix.
//!
//! The PLE block: the device holds its two projections as stored F16 for the dense matvec; the
//! host PLE feed (`ple_feed`) reads the norms and convolution from the pack itself, because
//! `ple::step` applies `1 + w` on the host and no device PLE block exists.
//!
//! The embedding stays off the device: the 1.18 GiB BF16 table is gathered one 5 KiB row per
//! token on the host (the S2 feed), which every other family does and which costs a page-cache
//! read; the card keeps the 1.18 GiB for K/V. Moving it to the card later is one more entry in
//! this list (`Form::Half`, read by `embed_bf16`).
//!
//! Family to onboard: a model of this layout gives its own recording walk and, if its norms or
//! vectors differ, its own [`form_of`] rules; the upload is shared.

// Staged: the Flash-Next CUDA step is the first caller.
#![cfg_attr(not(test), allow(dead_code))]

use crate::{
    EngError,
    family::{
        exl3_pack::{EMBED_NAME, Visit, VisitKind},
        flash_next::load::{exl3_pack::inspect_flash_pack_listing, model_shape},
    },
    weights::StreamedCheckpoint,
};

const LAYER_ROOT: &str = "model.language_model.layers.";

/// How the device holds one pack tensor, and the kernel that reads that form. A new tensor is
/// classified by its consumer's contract, not by its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family) enum Form {
    /// An EXL3 matrix under its prefix: trellis as stored, `suh` and `svh` in F32. Read by the EXL3
    /// projection kernels.
    Matrix,
    /// A plain tensor as stored (F16 or BF16). Read by the hyper-connection kernels
    /// (`resident/hyper.cu`: the kernel widens exactly and applies `1 + w` to `hc_norm` itself) and,
    /// for the two PLE projections, by the dense F16 matvec (`dense_gemv`).
    Half,
    /// A plain tensor widened to F32 as stored. Read by `router.cu` (router, shared-expert gate), the
    /// strided F32 router path (`in_proj_a`, `in_proj_b`), the DeltaNet kernels (`dt_bias`, conv,
    /// gated `norm`, which the host also uses raw) and `qsa_index.cu`, whose indexer norms take the
    /// stored zero-centered weight and add the 1 themselves (`(1.0f + w[d])`).
    Float,
    /// A zero-centered norm weight as `1 + w` in F32. Read by the vendored `attn_prep`, which
    /// multiplies the RMS-normalized head by the weight it is given (`q_norm`, `k_norm`).
    NormPlusOne,
    /// DeltaNet `A_log` as `-exp(A_log)` in F32, the decay the scan kernel reads.
    Decay,
}

/// One tensor of the trunk (for a matrix, its prefix) with the device bytes the walk counted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family) struct TrunkTensor {
    pub(in crate::family) name: String,
    pub(in crate::family) form: Form,
    pub(in crate::family) bytes: u64,
}

/// The role of a tensor: its name with the layer path removed.
fn role(name: &str) -> &str {
    name.strip_prefix(LAYER_ROOT)
        .and_then(|rest| rest.split_once('.'))
        .map_or(name, |(_, role)| role)
}

/// Matrices of the trunk by role. A routed expert is not here (the slot bank owns it).
const MATRICES: [&str; 12] = [
    "lm_head",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.indexer.index_qk_proj",
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
];

/// The tails of a hyper-connection site (the mixer has no `block_inject_weight`).
const HYPER_TAILS: [&str; 4] = [
    "hc_norm.weight",
    "input_mix_weight_down.weight",
    "input_mix_weight_up.weight",
    "block_inject_weight.weight",
];

/// The mixer that feeds `lm_head`, and the two sites of each layer, by role prefix.
const MIXER: &str = "model.language_model.hyper_connection_mixer.";
const SITES: [&str; 2] = ["attn_hyper_connection.", "mlp_hyper_connection."];

/// The PLE projections, the only PLE tensors the device holds.
const PLE_HALF: [&str; 2] = ["ple.key_proj.weight", "ple.value_proj.weight"];

fn is_hyper(name: &str, role: &str) -> bool {
    (name.starts_with(MIXER) || SITES.iter().any(|site| role.starts_with(site)))
        && HYPER_TAILS
            .iter()
            .any(|tail| name.ends_with(&format!(".{tail}")))
}

/// F32 tensors by role, as stored.
const FLOAT: [&str; 9] = [
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.dt_bias",
    "linear_attn.conv1d.weight",
    "linear_attn.norm.weight",
    "mlp.gate.weight",
    "mlp.shared_expert_gate.weight",
    "self_attn.indexer.q_layernorm.weight",
    "self_attn.indexer.k_layernorm.weight",
];

const NORM_PLUS_ONE: [&str; 2] = ["self_attn.q_norm.weight", "self_attn.k_norm.weight"];

/// The device form of a visited tensor: `Ok(None)` when the trunk does not hold it (a routed expert,
/// the embedding, a tensor a host feed reads), a refusal naming it when it is not in the tables, so
/// a new tensor of the walk cannot be widened by default into the wrong form.
fn form_of(visit: &Visit) -> Result<Option<Form>, EngError> {
    let name = visit.name.as_str();
    if name.contains(".mlp.experts.") || name == EMBED_NAME {
        return Ok(None);
    }
    let role = role(name);
    let form = match &visit.kind {
        VisitKind::Host { .. } => return Ok(None),
        VisitKind::Matrix { .. } if MATRICES.contains(&role) => Form::Matrix,
        VisitKind::Plain { keep: true, .. } if is_hyper(name, role) || PLE_HALF.contains(&role) => {
            Form::Half
        }
        VisitKind::Plain { keep: false, .. } if role == "linear_attn.A_log" => Form::Decay,
        VisitKind::Plain { keep: false, .. } if NORM_PLUS_ONE.contains(&role) => Form::NormPlusOne,
        VisitKind::Plain { keep: false, .. } if FLOAT.contains(&role) => Form::Float,
        _ => {
            return Err(EngError::Checkpoint(format!(
                "Flash-Next trunk: no device form for {name} ({:?}); add it to the tables of \
                 cuda/load.rs with its consumer kernel",
                visit.kind
            )));
        }
    };
    Ok(Some(form))
}

/// The layer a pack name belongs to, if it is under `model.language_model.layers.N.`.
fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix(LAYER_ROOT)?;
    rest.split_once('.')?.0.parse().ok()
}

/// The trunk tensors outside the layers and of each layer, in walk order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family) struct TrunkList {
    root: Vec<TrunkTensor>,
    layers: Vec<Vec<TrunkTensor>>,
    bytes: u64,
}

impl TrunkList {
    /// Sorts the visits of a walk over `layers` layers into the trunk. A visit under a layer
    /// outside the model is a refusal.
    pub(in crate::family) fn from_visits(
        visits: &[Visit],
        layers: usize,
    ) -> Result<Self, EngError> {
        let mut list = Self {
            root: Vec::new(),
            layers: vec![Vec::new(); layers],
            bytes: 0,
        };
        for visit in visits {
            let Some(form) = form_of(visit)? else {
                continue;
            };
            let tensor = TrunkTensor {
                name: visit.name.clone(),
                form,
                bytes: visit.bytes,
            };
            list.bytes += tensor.bytes;
            match layer_of(&tensor.name) {
                None => list.root.push(tensor),
                Some(layer) => list
                    .layers
                    .get_mut(layer)
                    .ok_or_else(|| {
                        EngError::Checkpoint(format!(
                            "Flash-Next trunk: {} is outside the model's {layers} layers",
                            tensor.name
                        ))
                    })?
                    .push(tensor),
            }
        }
        Ok(list)
    }

    /// Walks the pack's headers against `model_config` and lists the trunk. The sum of the trunk,
    /// the routed experts and the embedding must be the walk's device bytes.
    pub(in crate::family) fn new(
        source: &StreamedCheckpoint,
        model_config: &[u8],
    ) -> Result<Self, EngError> {
        let (_, kinds) = model_shape(model_config)?;
        let (report, visits) = inspect_flash_pack_listing(source, model_config)?;
        let list = Self::from_visits(&visits, kinds.len())?;
        let embedding: u64 = visits
            .iter()
            .filter(|visit| visit.name == EMBED_NAME)
            .map(|visit| visit.bytes)
            .sum();
        if list.bytes + report.expert_bytes + embedding != report.device_bytes {
            return Err(EngError::Checkpoint(format!(
                "Flash-Next trunk: {} trunk, {} expert and {embedding} embedding bytes do not make \
                 the walk's {}",
                list.bytes, report.expert_bytes, report.device_bytes
            )));
        }
        Ok(list)
    }

    /// Device bytes of the whole trunk.
    pub(in crate::family) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(in crate::family) fn layers(&self) -> usize {
        self.layers.len()
    }

    /// The tensors outside the layers: the mixer and `lm_head`.
    pub(in crate::family) fn root(&self) -> &[TrunkTensor] {
        &self.root
    }

    pub(in crate::family) fn layer(&self, layer: usize) -> Result<&[TrunkTensor], EngError> {
        self.layers.get(layer).map(Vec::as_slice).ok_or_else(|| {
            EngError::BadRequest(format!(
                "Flash-Next layer {layer} is outside the model's {} layers",
                self.layers.len()
            ))
        })
    }

    /// Every trunk tensor: the root, then each layer in order.
    pub(in crate::family) fn all(&self) -> impl Iterator<Item = &TrunkTensor> {
        self.root.iter().chain(self.layers.iter().flatten())
    }
}

/// The upload, for builds with the CUDA backend.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in crate::family) mod upload {
    use super::{Form, TrunkList, TrunkTensor};
    use crate::{
        EngError,
        family::{
            flash_next::moe::load_plan::FlashLoadPlan,
            project::cuda::{CudaWeight, CudaWeights},
            qwen36::{cuda::exl3_source::PackReader, exl3_tensors::decay_from_log},
        },
    };

    /// Uploads one tensor through the reader's primitive for its form. Shapes of plain tensors
    /// come from the stored headers, which the walk checked against `config.json`.
    fn upload(reader: &PackReader<'_>, tensor: &TrunkTensor) -> Result<CudaWeight, EngError> {
        let name = tensor.name.as_str();
        if tensor.form == Form::Matrix {
            let matrix = reader.matrix(name)?;
            return reader.upload(name.to_owned(), &matrix);
        }
        let (_, shape) = reader.checkpoint.source.stored(name).ok_or_else(|| {
            EngError::Checkpoint(format!("Flash-Next trunk: missing tensor {name}"))
        })?;
        let owned = name.to_owned();
        match tensor.form {
            Form::Matrix => unreachable!("returned above"),
            Form::Half => reader.table(owned, name, &shape),
            Form::Float => reader.vector(owned, name, &shape, Ok),
            Form::Decay => reader.vector(owned, name, &shape, |log| decay_from_log(&log)),
            Form::NormPlusOne => {
                let values = shape.iter().map(|&dim| dim as usize).product();
                reader.norm(owned, name, values)
            }
        }
    }

    fn upload_all(
        reader: &PackReader<'_>,
        tensors: &[TrunkTensor],
    ) -> Result<Vec<CudaWeight>, EngError> {
        tensors
            .iter()
            .map(|tensor| upload(reader, tensor))
            .collect()
    }

    /// The tensors outside the layers (the mixer and `lm_head`).
    pub(in crate::family) fn root_weights(
        reader: &PackReader<'_>,
        list: &TrunkList,
    ) -> Result<Vec<CudaWeight>, EngError> {
        upload_all(reader, list.root())
    }

    /// The trunk tensors of one layer.
    pub(in crate::family) fn layer_weights(
        reader: &PackReader<'_>,
        list: &TrunkList,
        layer: usize,
    ) -> Result<Vec<CudaWeight>, EngError> {
        upload_all(reader, list.layer(layer)?)
    }

    /// The whole trunk. The list must hold the plan's trunk bytes
    /// (`FlashLoadPlan::trunk_bytes`); the load is admitted against them before the first upload
    /// and the device must end up holding exactly that many.
    pub(in crate::family) fn trunk_weights(
        reader: &PackReader<'_>,
        list: &TrunkList,
        plan: &FlashLoadPlan,
    ) -> Result<CudaWeights, EngError> {
        let plan_trunk_bytes = plan.trunk_bytes;
        if list.bytes() != plan_trunk_bytes {
            return Err(EngError::MemoryPlan(format!(
                "Flash-Next trunk list holds {} device bytes, the plan counted {plan_trunk_bytes}",
                list.bytes()
            )));
        }
        reader.admit(plan_trunk_bytes)?;
        let mut weights = root_weights(reader, list)?;
        for layer in 0..list.layers() {
            weights.extend(layer_weights(reader, list, layer)?);
        }
        let loaded = CudaWeights::from_exl3(weights)?;
        let held = loaded.device_bytes() as u64;
        if held != plan_trunk_bytes {
            return Err(EngError::MemoryPlan(format!(
                "Flash-Next trunk holds {held} device bytes, the plan counted {plan_trunk_bytes}"
            )));
        }
        Ok(loaded)
    }
}
