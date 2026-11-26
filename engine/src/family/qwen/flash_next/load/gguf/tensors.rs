//! The tensors of a `qwen4exp` GGUF mapped to the weight structures the Flash-Next executor
//! consumes.
//!
//! [`TensorMap::build`] derives every tensor the model needs from the configuration (layer
//! kinds, widths, counts), looks each up by its GGUF name, and checks its dimensions. A missing
//! tensor or a wrong shape refuses the load naming the tensor. A tensor of the file that no
//! role claims is reported in [`TensorMap::unmapped`] and left unread. Every role has the form
//! the device holds it in ([`Form`], the counterpart of the EXL3 trunk's forms) and the
//! convention of the stored values ([`Convention`]) relative to what the EXL3 pack stores.
//!
//! The GGUF layout differs from the EXL3 pack in four places the executor must honour:
//! - Zero-centred norm gammas are stored as `1 + w`: the hyper-connection norms, the indexer
//!   norms, the PLE norms and the attention `q_norm` and `k_norm`. The EXL3 pack stores `w`.
//!   The kernels that add the 1 themselves (hyper-connection, indexer, the host PLE step) take
//!   `stored - 1`; the attention prep reads `1 + w` and takes the stored value as it is.
//! - The DeltaNet value heads are in tiled order (key head `h % key_heads`), where the pack
//!   keeps the grouped order. The scan kernel has both orders (`GdnKeyHeads`).
//! - `A_log` is stored as `-exp(A_log)`, the decay form the scan reads.
//! - The indexer projection is two tensors, queries then the key; the pack has one fused matrix.

use std::collections::HashSet;

use super::{
    super::Dims,
    config::{GgufConfig, is_float},
};
use crate::{
    EngError,
    family::recipe::{Global, Role as Wanted, Site, SiteTensor},
    weights::{GgufHeader, GgufType},
};

/// What a tensor of the file serves: a role of one layer, or a role outside the layers, named as
/// the recipe names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::family::qwen::flash_next) enum Slot {
    Layer(Wanted),
    Global(Global),
}

impl Slot {
    /// One tensor of a hyper-connection site: a site of a layer, or the final site that feeds
    /// the head.
    pub(in crate::family::qwen::flash_next) fn hyper(site: Site, part: SiteTensor) -> Self {
        match site {
            Site::Head => Self::Global(Global::HeadSite(part)),
            Site::Mixer | Site::Ffn => Self::Layer(Wanted::Hyper(site, part)),
        }
    }
}

impl From<Wanted> for Slot {
    fn from(role: Wanted) -> Self {
        Self::Layer(role)
    }
}

impl From<Global> for Slot {
    fn from(role: Global) -> Self {
        Self::Global(role)
    }
}

/// Which memory a role's bytes serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::family::qwen::flash_next) enum Place {
    /// Uploaded once and resident on the card for the run.
    Device,
    /// Read on the host: the embedding rows of each token and the PLE vectors.
    Host,
    /// The routed experts, streamed from the file through the expert tiers.
    Experts,
    /// The n-gram table, read row by row from the file.
    Ngram,
}

/// How the device holds a tensor (the counterpart of the EXL3 trunk's `Form`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::qwen::flash_next) enum Form {
    /// The GGUF blocks as stored; the matvec kernel of the tensor's type reads them.
    Blocks,
    /// Widened to F16 at upload, for the kernels that read half-precision weights. The staged
    /// start keeps a Q8_0 hyper mixing matrix of this form as stored instead.
    Half,
    /// Widened to F32 at upload.
    Float,
    /// A norm gamma as `1 + w` in F32.
    NormPlusOne,
    /// `-exp(A_log)` in F32, the decay the scan reads.
    Decay,
    /// Not held on the device.
    None,
}

/// How the stored values relate to the EXL3 pack's stored values for the same role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::qwen::flash_next) enum Convention {
    AsStored,
    /// Stored as `1 + w`; the EXL3 pack stores `w`. A consumer that adds the 1 itself
    /// subtracts it at load.
    GammaOnePlusW,
}

/// One tensor of the file claimed by a role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::qwen::flash_next) struct Mapped {
    pub(in crate::family::qwen::flash_next) role: Slot,
    pub(in crate::family::qwen::flash_next) layer: Option<usize>,
    pub(in crate::family::qwen::flash_next) name: String,
    pub(in crate::family::qwen::flash_next) ty: GgufType,
    pub(in crate::family::qwen::flash_next) dims: Vec<u64>,
    pub(in crate::family::qwen::flash_next) bytes: u64,
    pub(in crate::family::qwen::flash_next) shard: usize,
    pub(in crate::family::qwen::flash_next) place: Place,
    pub(in crate::family::qwen::flash_next) form: Form,
    pub(in crate::family::qwen::flash_next) convention: Convention,
}

impl Mapped {
    /// Tensor-relative stored bytes for contiguous output rows of a matrix or expert.
    /// This plans file reads only; it does not upload or execute the resulting blocks.
    pub(in super::super::super) fn row_span(
        &self,
        expert: Option<u64>,
        rows: std::ops::Range<u64>,
    ) -> Result<(u64, usize), EngError> {
        let invalid = || EngError::BadRequest(format!("invalid row slice for {}", self.name));
        let (k, outputs, expert) = match (self.dims.as_slice(), expert) {
            ([k, outputs], None) => (*k, *outputs, 0),
            ([k, outputs, experts], Some(expert)) if expert < *experts => (*k, *outputs, expert),
            _ => return Err(invalid()),
        };
        let block = self.ty.block_elems();
        if k == 0 || !k.is_multiple_of(block) || rows.start >= rows.end || rows.end > outputs {
            return Err(invalid());
        }
        let stride = (k / block)
            .checked_mul(self.ty.block_bytes())
            .ok_or_else(invalid)?;
        let begin = expert
            .checked_mul(outputs)
            .and_then(|n| n.checked_add(rows.start))
            .and_then(|n| n.checked_mul(stride))
            .ok_or_else(invalid)?;
        let bytes = (rows.end - rows.start)
            .checked_mul(stride)
            .filter(|bytes| {
                begin
                    .checked_add(*bytes)
                    .is_some_and(|end| end <= self.bytes)
            })
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(invalid)?;
        Ok((begin, bytes))
    }

    /// Bytes this tensor takes on the card in its device form (zero off the card): what the
    /// trunk loader uploads, a `Half` matrix widened to F16.
    pub(in crate::family::qwen::flash_next) fn device_bytes(&self) -> u64 {
        let elements: u64 = self.dims.iter().product();
        match self.form {
            Form::Blocks => self.bytes,
            Form::Half => elements * 2,
            Form::Float | Form::NormPlusOne | Form::Decay => elements * 4,
            Form::None => 0,
        }
    }

    /// The same in the form the staged start uploads it in: a Q8_0 mixing matrix of a hyper
    /// site stays as stored, 34 bytes a block.
    pub(in crate::family::qwen::flash_next) fn staged_device_bytes(&self) -> u64 {
        if self.form == Form::Half && self.q80_stays_stored() {
            self.bytes
        } else {
            self.device_bytes()
        }
    }

    /// A hyper mixing matrix (`down`, `up`) of a layer or of the final site stored as Q8_0, which
    /// [`crate::family::qwen::flash_next::load::gguf::GgufModel::upload_hyper`] keeps as stored.
    pub(in crate::family::qwen::flash_next) fn q80_stays_stored(&self) -> bool {
        let part = match self.role {
            Slot::Layer(Wanted::Hyper(_, part)) | Slot::Global(Global::HeadSite(part)) => part,
            _ => return false,
        };
        self.ty == GgufType::Q80 && part.q80_stays_stored()
    }

    /// Whether a matvec kernel of this tensor's type reads it (block formats only).
    fn needs_kernel(&self) -> bool {
        matches!(self.form, Form::Blocks) || self.place == Place::Experts
    }
}

/// A tensor type some matrix of the model has and no CUDA decode kernel reads yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::qwen::flash_next) struct KernelGap {
    pub(in crate::family::qwen::flash_next) ty: GgufType,
    pub(in crate::family::qwen::flash_next) tensors: usize,
    pub(in crate::family::qwen::flash_next) bytes: u64,
    pub(in crate::family::qwen::flash_next) first: String,
}

/// What a role requires of its tensor's type.
#[derive(Clone, Copy)]
enum Kind {
    /// A weight matrix of any block or float type.
    Matrix,
    /// Floating-point values (F32, F16 or BF16).
    Vector,
}

struct Expect {
    role: Slot,
    layer: Option<usize>,
    names: Vec<String>,
    dims: Vec<u64>,
    kind: Kind,
}

fn site_prefix(site: Site, layer: Option<usize>) -> String {
    match (site, layer) {
        (Site::Head, _) => "output_hc".to_string(),
        (Site::Mixer, Some(layer)) => format!("blk.{layer}.hc_attn"),
        (Site::Ffn, Some(layer)) => format!("blk.{layer}.hc_ffn"),
        _ => unreachable!("a layer site has a layer"),
    }
}

/// Every tensor the model needs, in the order of the executor's structures.
fn expected(dims: &Dims, rows_needed: u64) -> Vec<Expect> {
    let d = |v: u32| u64::from(v);
    let (hidden, width, rank, branches) = (
        d(dims.hidden),
        d(dims.width),
        d(dims.rank),
        d(dims.branches),
    );
    let mut out = Vec::new();
    let mut add = |role, layer: Option<usize>, names: Vec<String>, shape: Vec<u64>, kind| {
        out.push(Expect {
            role,
            layer,
            names,
            dims: shape,
            kind,
        });
    };
    let one = |name: &str| vec![name.to_string()];
    add(
        Slot::Global(Global::Embedding),
        None,
        one("token_embd.weight"),
        vec![hidden, d(dims.vocab)],
        Kind::Matrix,
    );
    // A model without its own output matrix reuses the embedding.
    add(
        Slot::Global(Global::Head),
        None,
        vec!["output.weight".into(), "token_embd.weight".into()],
        vec![hidden, d(dims.vocab)],
        Kind::Matrix,
    );
    let hyper = |add: &mut dyn FnMut(Slot, Option<usize>, Vec<String>, Vec<u64>, Kind),
                 site: Site,
                 layer: Option<usize>| {
        let prefix = site_prefix(site, layer);
        add(
            Slot::hyper(site, SiteTensor::Norm),
            layer,
            vec![format!("{prefix}_norm.weight")],
            vec![width],
            Kind::Vector,
        );
        add(
            Slot::hyper(site, SiteTensor::Down),
            layer,
            vec![format!("{prefix}_down.weight")],
            vec![width, rank],
            Kind::Matrix,
        );
        add(
            Slot::hyper(site, SiteTensor::Up),
            layer,
            vec![format!("{prefix}_up.weight")],
            vec![rank, width],
            Kind::Matrix,
        );
        if site != Site::Head {
            add(
                Slot::hyper(site, SiteTensor::Inject),
                layer,
                vec![format!("{prefix}_inject.weight")],
                vec![width, branches],
                Kind::Matrix,
            );
        }
    };
    hyper(&mut add, Site::Head, None);
    let key_width = d(dims.key_heads) * d(dims.key_dim);
    let value_width = d(dims.value_heads) * d(dims.value_dim);
    let conv_dim = 2 * key_width + value_width;
    let q_width = d(dims.heads_q) * d(dims.head_dim);
    let kv_width = d(dims.heads_kv) * d(dims.head_dim);
    let (inter, shared, experts) = (d(dims.moe_inter), d(dims.shared_inter), d(dims.experts));
    for (layer, &attention) in dims.kinds.iter().enumerate() {
        let at = Some(layer);
        let blk = |suffix: &str| vec![format!("blk.{layer}.{suffix}")];
        hyper(&mut add, Site::Mixer, at);
        if attention {
            add(
                Slot::Layer(Wanted::QProj),
                at,
                blk("attn_q.weight"),
                vec![hidden, 2 * q_width],
                Kind::Matrix,
            );
            add(
                Slot::Layer(Wanted::KProj),
                at,
                blk("attn_k.weight"),
                vec![hidden, kv_width],
                Kind::Matrix,
            );
            add(
                Slot::Layer(Wanted::VProj),
                at,
                blk("attn_v.weight"),
                vec![hidden, kv_width],
                Kind::Matrix,
            );
            add(
                Slot::Layer(Wanted::OProj),
                at,
                blk("attn_output.weight"),
                vec![q_width, hidden],
                Kind::Matrix,
            );
            add(
                Slot::Layer(Wanted::IndexProj),
                at,
                blk("indexer.q_proj.weight"),
                vec![hidden, d(dims.idx_heads) * d(dims.idx_dim)],
                Kind::Matrix,
            );
            add(
                Slot::Layer(Wanted::IndexKey),
                at,
                blk("indexer.k_proj.weight"),
                vec![hidden, d(dims.idx_dim)],
                Kind::Matrix,
            );
            add(
                Slot::Layer(Wanted::QNorm),
                at,
                blk("attn_q_norm.weight"),
                vec![d(dims.head_dim)],
                Kind::Vector,
            );
            add(
                Slot::Layer(Wanted::KNorm),
                at,
                blk("attn_k_norm.weight"),
                vec![d(dims.head_dim)],
                Kind::Vector,
            );
            add(
                Slot::Layer(Wanted::IndexQNorm),
                at,
                blk("indexer.q_norm.weight"),
                vec![d(dims.idx_dim)],
                Kind::Vector,
            );
            add(
                Slot::Layer(Wanted::IndexKNorm),
                at,
                blk("indexer.k_norm.weight"),
                vec![d(dims.idx_dim)],
                Kind::Vector,
            );
        } else {
            add(
                Slot::Layer(Wanted::DeltaQkv),
                at,
                blk("attn_qkv.weight"),
                vec![hidden, conv_dim],
                Kind::Matrix,
            );
            add(
                Slot::Layer(Wanted::DeltaGate),
                at,
                blk("attn_gate.weight"),
                vec![hidden, value_width],
                Kind::Matrix,
            );
            add(
                Slot::Layer(Wanted::DeltaBeta),
                at,
                blk("ssm_beta.weight"),
                vec![hidden, d(dims.value_heads)],
                Kind::Matrix,
            );
            add(
                Slot::Layer(Wanted::DeltaAlpha),
                at,
                blk("ssm_alpha.weight"),
                vec![hidden, d(dims.value_heads)],
                Kind::Matrix,
            );
            add(
                Slot::Layer(Wanted::DeltaOut),
                at,
                blk("ssm_out.weight"),
                vec![value_width, hidden],
                Kind::Matrix,
            );
            add(
                Slot::Layer(Wanted::DeltaConv),
                at,
                blk("ssm_conv1d.weight"),
                vec![d(dims.conv_kernel), conv_dim],
                Kind::Vector,
            );
            add(
                Slot::Layer(Wanted::DeltaDecay),
                at,
                blk("ssm_a"),
                vec![d(dims.value_heads)],
                Kind::Vector,
            );
            add(
                Slot::Layer(Wanted::DeltaDtBias),
                at,
                blk("ssm_dt.bias"),
                vec![d(dims.value_heads)],
                Kind::Vector,
            );
            add(
                Slot::Layer(Wanted::DeltaNorm),
                at,
                blk("ssm_norm.weight"),
                vec![d(dims.value_dim)],
                Kind::Vector,
            );
        }
        if layer == dims.ple_layer {
            let embed = d(dims.ple_embed);
            add(
                Slot::Layer(Wanted::PleKey),
                at,
                blk("ple_key.weight"),
                vec![embed, width],
                Kind::Matrix,
            );
            add(
                Slot::Layer(Wanted::PleValue),
                at,
                blk("ple_value.weight"),
                vec![embed, hidden],
                Kind::Matrix,
            );
            add(
                Slot::Layer(Wanted::PleNormKey),
                at,
                blk("ple_norm_key.weight"),
                vec![width],
                Kind::Vector,
            );
            add(
                Slot::Layer(Wanted::PleNormQuery),
                at,
                blk("ple_norm_query.weight"),
                vec![width],
                Kind::Vector,
            );
            add(
                Slot::Layer(Wanted::PleNormConv),
                at,
                blk("ple_norm_conv.weight"),
                vec![width],
                Kind::Vector,
            );
            add(
                Slot::Layer(Wanted::PleConv),
                at,
                blk("ple_conv1d.weight"),
                vec![d(dims.ple_conv), width],
                Kind::Vector,
            );
        }
        hyper(&mut add, Site::Ffn, at);
        add(
            Slot::Layer(Wanted::Router),
            at,
            blk("ffn_gate_inp.weight"),
            vec![hidden, experts],
            Kind::Vector,
        );
        add(
            Slot::Layer(Wanted::SharedRouterGate),
            at,
            blk("ffn_gate_inp_shexp.weight"),
            vec![hidden],
            Kind::Vector,
        );
        add(
            Slot::Layer(Wanted::SharedGate),
            at,
            blk("ffn_gate_shexp.weight"),
            vec![hidden, shared],
            Kind::Matrix,
        );
        add(
            Slot::Layer(Wanted::SharedUp),
            at,
            blk("ffn_up_shexp.weight"),
            vec![hidden, shared],
            Kind::Matrix,
        );
        add(
            Slot::Layer(Wanted::SharedDown),
            at,
            blk("ffn_down_shexp.weight"),
            vec![shared, hidden],
            Kind::Matrix,
        );
        add(
            Slot::Layer(Wanted::ExpertGate),
            at,
            blk("ffn_gate_exps.weight"),
            vec![hidden, inter, experts],
            Kind::Matrix,
        );
        add(
            Slot::Layer(Wanted::ExpertUp),
            at,
            blk("ffn_up_exps.weight"),
            vec![hidden, inter, experts],
            Kind::Matrix,
        );
        add(
            Slot::Layer(Wanted::ExpertDown),
            at,
            blk("ffn_down_exps.weight"),
            vec![inter, hidden, experts],
            Kind::Matrix,
        );
    }
    add(
        Slot::Global(Global::NgramTable),
        None,
        vec!["per_layer_token_embd.weight".into()],
        vec![d(dims.ple_head_dim), rows_needed],
        Kind::Matrix,
    );
    out
}

/// The place, device form and value convention of a role.
fn traits_of(role: Slot) -> (Place, Form, Convention) {
    use Convention::{AsStored, GammaOnePlusW};
    use Form::{Blocks, Decay, Float, Half, NormPlusOne};
    use Place::{Device, Experts, Host, Ngram};
    match role {
        Slot::Global(Global::Embedding) => (Host, Form::None, AsStored),
        Slot::Global(Global::Head)
        | Slot::Layer(Wanted::DeltaQkv)
        | Slot::Layer(Wanted::DeltaGate)
        | Slot::Layer(Wanted::DeltaOut)
        | Slot::Layer(Wanted::QProj)
        | Slot::Layer(Wanted::KProj)
        | Slot::Layer(Wanted::VProj)
        | Slot::Layer(Wanted::OProj)
        | Slot::Layer(Wanted::SharedGate)
        | Slot::Layer(Wanted::SharedUp)
        | Slot::Layer(Wanted::SharedDown)
        | Slot::Layer(Wanted::PleKey)
        | Slot::Layer(Wanted::PleValue) => (Device, Blocks, AsStored),
        Slot::Layer(Wanted::Hyper(_, SiteTensor::Down | SiteTensor::Up | SiteTensor::Inject))
        | Slot::Global(Global::HeadSite(SiteTensor::Down | SiteTensor::Up)) => {
            (Device, Half, AsStored)
        }
        Slot::Layer(Wanted::IndexProj) | Slot::Layer(Wanted::IndexKey) => (Device, Half, AsStored),
        Slot::Layer(Wanted::Hyper(_, SiteTensor::Norm))
        | Slot::Global(Global::HeadSite(SiteTensor::Norm)) => (Device, Float, GammaOnePlusW),
        Slot::Layer(Wanted::IndexQNorm) | Slot::Layer(Wanted::IndexKNorm) => {
            (Device, Float, GammaOnePlusW)
        }
        Slot::Layer(Wanted::QNorm) | Slot::Layer(Wanted::KNorm) => {
            (Device, NormPlusOne, GammaOnePlusW)
        }
        Slot::Layer(Wanted::DeltaDecay) => (Device, Decay, AsStored),
        Slot::Layer(Wanted::DeltaBeta)
        | Slot::Layer(Wanted::DeltaAlpha)
        | Slot::Layer(Wanted::DeltaConv)
        | Slot::Layer(Wanted::DeltaDtBias)
        | Slot::Layer(Wanted::DeltaNorm)
        | Slot::Layer(Wanted::Router)
        | Slot::Layer(Wanted::SharedRouterGate) => (Device, Float, AsStored),
        Slot::Layer(Wanted::PleNormKey)
        | Slot::Layer(Wanted::PleNormQuery)
        | Slot::Layer(Wanted::PleNormConv) => (Host, Form::None, GammaOnePlusW),
        Slot::Layer(Wanted::PleConv) => (Host, Form::None, AsStored),
        Slot::Layer(Wanted::ExpertGate)
        | Slot::Layer(Wanted::ExpertUp)
        | Slot::Layer(Wanted::ExpertDown) => (Experts, Form::None, AsStored),
        Slot::Global(Global::NgramTable) => (Ngram, Form::None, AsStored),
        // The roles of other architectures: the file's inventory never lists them.
        _ => unreachable!("no tensor of this role is expected"),
    }
}

/// One tensor of the inventory: what [`TensorMap::build`] looks for.
#[cfg(test)]
pub(super) struct InventoryItem {
    pub(super) role: Slot,
    pub(super) layer: Option<usize>,
    pub(super) name: String,
    pub(super) dims: Vec<u64>,
    pub(super) vector: bool,
}

/// Every tensor the model needs by its first-choice name (the tests write files from it).
#[cfg(test)]
pub(super) fn inventory(config: &GgufConfig) -> Vec<InventoryItem> {
    expected(&config.dims, config.ngram_rows_needed)
        .into_iter()
        .map(|want| InventoryItem {
            role: want.role,
            layer: want.layer,
            name: want.names[0].clone(),
            dims: want.dims,
            vector: matches!(want.kind, Kind::Vector),
        })
        .collect()
}

fn mismatch(detail: String) -> EngError {
    EngError::Checkpoint(format!("Flash-Next GGUF tensors: {detail}"))
}

/// The tensors of a file mapped to roles.
#[derive(Debug)]
pub(in crate::family::qwen::flash_next) struct TensorMap {
    entries: Vec<Mapped>,
    unmapped: Vec<String>,
    total: usize,
    mapped_names: usize,
}

impl TensorMap {
    /// Maps every tensor of `header` that a role of the model claims. A role whose tensor is
    /// missing, has other dimensions, or has a non-float type where floats are required
    /// refuses the whole map; the error lists up to five such tensors and the total.
    pub(in crate::family::qwen::flash_next) fn build(
        header: &GgufHeader,
        config: &GgufConfig,
    ) -> Result<Self, EngError> {
        let dims = &config.dims;
        let needed = expected(dims, config.ngram_rows_needed);
        let mut entries = Vec::with_capacity(needed.len());
        let mut problems = Vec::new();
        let mut claimed: HashSet<&str> = HashSet::new();
        for want in &needed {
            let found = want
                .names
                .iter()
                .find_map(|name| header.tensor(name).map(|tensor| (name, tensor)));
            let Some((name, tensor)) = found else {
                problems.push(format!("{} is missing", want.names[0]));
                continue;
            };
            let is_ngram = want.role == Slot::Global(Global::NgramTable);
            // The n-gram table may hold more rows than the heads address.
            let dims_agree = if is_ngram {
                tensor.dims().len() == 2
                    && tensor.dims()[0] == want.dims[0]
                    && tensor.dims()[1] >= want.dims[1]
            } else {
                tensor.dims() == want.dims.as_slice()
            };
            if !dims_agree {
                problems.push(format!(
                    "{name} has dimensions {:?}, the model needs {:?}",
                    tensor.dims(),
                    want.dims
                ));
                continue;
            }
            if matches!(want.kind, Kind::Vector) && !is_float(tensor.ggml_type()) {
                problems.push(format!(
                    "{name} is {}, the model needs F32, F16 or BF16 values",
                    tensor.ggml_type().name()
                ));
                continue;
            }
            claimed.insert(name.as_str());
            let (place, form, convention) = traits_of(want.role);
            entries.push(Mapped {
                role: want.role,
                layer: want.layer,
                name: name.clone(),
                ty: tensor.ggml_type(),
                dims: tensor.dims().to_vec(),
                bytes: tensor.byte_len(),
                shard: tensor.shard(),
                place,
                form,
                convention,
            });
        }
        if !problems.is_empty() {
            let shown = problems
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ");
            return Err(mismatch(format!(
                "{} of the model's {} tensors are missing or do not fit: {shown}",
                problems.len(),
                needed.len()
            )));
        }
        let unmapped = header
            .tensors()
            .iter()
            .map(|tensor| tensor.name())
            .filter(|name| !claimed.contains(name))
            .map(str::to_owned)
            .collect();
        Ok(Self {
            entries,
            unmapped,
            total: header.tensors().len(),
            mapped_names: claimed.len(),
        })
    }

    /// Every claimed tensor, in the order of the executor's structures.
    pub(in crate::family::qwen::flash_next) fn entries(&self) -> &[Mapped] {
        &self.entries
    }

    /// Tensors in the file.
    pub(in crate::family::qwen::flash_next) fn total(&self) -> usize {
        self.total
    }

    /// Tensors of the file claimed by a role (a tied head and embedding count once).
    pub(in crate::family::qwen::flash_next) fn mapped(&self) -> usize {
        self.mapped_names
    }

    /// Names of file tensors no role claims.
    pub(in crate::family::qwen::flash_next) fn unmapped(&self) -> &[String] {
        &self.unmapped
    }

    /// The entry of `role` in `layer` (`None` for roles outside the layers).
    pub(in crate::family::qwen::flash_next) fn get(
        &self,
        role: impl Into<Slot>,
        layer: Option<usize>,
    ) -> Option<&Mapped> {
        let role = role.into();
        self.entries
            .iter()
            .find(|entry| entry.role == role && entry.layer == layer)
    }

    /// File bytes of the tensors served from `place`.
    pub(in crate::family::qwen::flash_next) fn file_bytes(&self, place: Place) -> u64 {
        self.entries
            .iter()
            .filter(|entry| entry.place == place)
            .map(|entry| entry.bytes)
            .sum()
    }

    /// Device bytes of the resident trunk in the forms the executor holds it in.
    pub(in crate::family::qwen::flash_next) fn device_bytes(&self) -> u64 {
        self.entries.iter().map(Mapped::device_bytes).sum()
    }

    /// The same in the forms the staged start holds the trunk in.
    pub(in crate::family::qwen::flash_next) fn staged_device_bytes(&self) -> u64 {
        self.entries.iter().map(Mapped::staged_device_bytes).sum()
    }

    /// Maximum input width of mapped quantized device matrices and routed expert stacks.
    /// This bounds one serialized native activation row, not concurrent or prefill scratch.
    /// Returns zero when no mapped quantized projection needs a native kernel.
    pub(in super::super::super) fn max_quantized_input_width(&self) -> Result<usize, EngError> {
        self.entries
            .iter()
            .filter(|entry| entry.needs_kernel() && !is_float(entry.ty))
            .try_fold(0usize, |max, entry| {
                let k = entry
                    .dims
                    .first()
                    .copied()
                    .and_then(|k| usize::try_from(k).ok())
                    .filter(|k| *k != 0 && k.is_multiple_of(32))
                    .ok_or_else(|| {
                        mismatch(format!("{} has an invalid native input width", entry.name))
                    })?;
                Ok(max.max(k))
            })
    }

    /// The tensor types that matrices of the model have and `supported` does not accept,
    /// each with its tensor count, bytes and first tensor, in order of first appearance.
    pub(in crate::family::qwen::flash_next) fn kernel_gaps(
        &self,
        supported: impl Fn(Place, GgufType) -> bool,
    ) -> Vec<KernelGap> {
        let mut gaps: Vec<KernelGap> = Vec::new();
        for entry in self.entries.iter().filter(|e| e.needs_kernel()) {
            if is_float(entry.ty) || supported(entry.place, entry.ty) {
                continue;
            }
            match gaps.iter_mut().find(|gap| gap.ty == entry.ty) {
                Some(gap) => {
                    gap.tensors += 1;
                    gap.bytes += entry.bytes;
                }
                None => gaps.push(KernelGap {
                    ty: entry.ty,
                    tensors: 1,
                    bytes: entry.bytes,
                    first: entry.name.clone(),
                }),
            }
        }
        gaps
    }
}

/// Whether the in-tree CUDA kernels read a matrix of `ty` served from `place` as stored blocks:
/// the dense matvec kernels for the trunk, the expert kernels for the routed experts. Every type
/// of the shared native list has both (`GgufType::has_native_kernel`).
pub(in crate::family::qwen::flash_next) fn has_decode_kernel(_place: Place, ty: GgufType) -> bool {
    ty.has_native_kernel()
}

#[cfg(test)]
mod width_tests {
    use super::*;

    fn entry(place: Place, form: Form, ty: GgufType, k: u64) -> Mapped {
        Mapped {
            role: Global::Embedding.into(),
            layer: None,
            name: "test".into(),
            ty,
            dims: vec![k, 4],
            bytes: 0,
            shard: 0,
            place,
            form,
            convention: Convention::AsStored,
        }
    }

    fn map(entries: Vec<Mapped>) -> TensorMap {
        TensorMap {
            entries,
            unmapped: vec![],
            total: 0,
            mapped_names: 0,
        }
    }

    #[test]
    fn width_includes_experts_and_excludes_host_and_float_weights() {
        let entries = vec![
            entry(Place::Device, Form::Blocks, GgufType::Q80, 256),
            entry(Place::Experts, Form::None, GgufType::Q80, 1024),
            entry(Place::Host, Form::None, GgufType::Q80, 4096),
            entry(Place::Device, Form::Blocks, GgufType::F32, 2048),
        ];
        assert_eq!(map(entries).max_quantized_input_width().unwrap(), 1024);
        assert_eq!(map(vec![]).max_quantized_input_width().unwrap(), 0);
        assert!(
            map(vec![entry(Place::Device, Form::Blocks, GgufType::Q80, 33)])
                .max_quantized_input_width()
                .is_err()
        );
    }
}

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod cuda_tests {
    use super::*;
    use crate::kernels::cuda::resident::matvec_q::QuantType;

    /// Every matvec kernel format.
    const ALL_KERNELS: [QuantType; 18] = [
        QuantType::Q4K,
        QuantType::Q50,
        QuantType::Q6K,
        QuantType::Q80,
        QuantType::Q5K,
        QuantType::Q40,
        QuantType::Q3K,
        QuantType::Iq4Xs,
        QuantType::Iq4Nl,
        QuantType::Iq2S,
        QuantType::Iq3Xxs,
        QuantType::Iq2Xxs,
        QuantType::Q20,
        QuantType::Iq2Xs,
        QuantType::Iq3S,
        QuantType::Q2K,
        QuantType::Iq1S,
        QuantType::Iq1M,
    ];

    /// The format layer's list of natively served types, the plan's block table and the kernel
    /// formats are one set with one geometry.
    #[test]
    fn the_native_type_list_is_the_kernel_formats() {
        let mut from_kernels: Vec<u32> = ALL_KERNELS.map(|q| q.gguf().id()).to_vec();
        let mut listed: Vec<u32> = GgufType::native_types().iter().map(|ty| ty.id()).collect();
        from_kernels.sort_unstable();
        listed.sort_unstable();
        assert_eq!(listed, from_kernels);
        for quant in ALL_KERNELS {
            let ty = quant.gguf();
            assert_eq!(ty.block_elems() as usize, quant.block_size(), "{ty:?}");
            assert_eq!(ty.block_bytes() as usize, quant.block_bytes(), "{ty:?}");
            assert!(has_decode_kernel(Place::Experts, ty), "{ty:?}");
            assert!(has_decode_kernel(Place::Device, ty), "{ty:?}");
        }
    }
}
