//! The stock-pack tensor names of Qwen3.6 and a header-only inspection of a pack against the
//! model shape.
//!
//! [`inspect_pack`] reads no payload: it walks every tensor the loader will read, checks its
//! stored dtype and shape against the model, requires the routed experts of a projection to
//! share one rate (a bank has one stride), and sums the device bytes the load will hold. A pack
//! of another size, rate, codebook or layout is refused here, before the first upload, and the
//! same walk runs on a CPU host for any pack on disk.

use crate::{
    EngError,
    family::{AttentionKind, qwen36::host::HostShape},
    tensor::Dtype,
    weights::StreamedCheckpoint,
};

pub(in crate::family) use crate::family::exl3_pack::{
    EMBED_NAME, HEAD_PREFIX, Inspector, Marker, MoeBlock, NORM_NAME, PackReport, Root, layer_name,
};
#[cfg(test)]
pub(in crate::family) use crate::family::exl3_pack::{
    ExpertPart, expert_name_in, router_name_in, shared_gate_name_in, shared_name_in,
};

/// The router weight of a layer of the model.
#[cfg(test)]
pub(in crate::family) fn router_name(layer: usize) -> String {
    router_name_in(Root::Layers, layer)
}

/// The sigmoid gate of the shared expert of a layer of the model.
#[cfg(test)]
pub(super) fn shared_gate_name(layer: usize) -> String {
    shared_gate_name_in(Root::Layers, layer)
}

/// One projection of one routed expert of a layer of the model.
#[cfg(test)]
pub(in crate::family) fn expert_name(layer: usize, expert: usize, part: ExpertPart) -> String {
    expert_name_in(Root::Layers, layer, expert, part)
}

/// One projection of the shared expert of a layer of the model.
#[cfg(test)]
pub(in crate::family) fn shared_name(layer: usize, part: ExpertPart) -> String {
    shared_name_in(Root::Layers, layer, part)
}

fn refuse(detail: String) -> EngError {
    EngError::Checkpoint(format!("EXL3 pack: {detail}"))
}

pub(super) fn inspect_pack(
    source: &StreamedCheckpoint,
    shape: &HostShape,
) -> Result<PackReport, EngError> {
    inspect_pack_split(source, shape).map(|split| split.report)
}

/// The pack walk plus its expert/dense split: `routed_bytes` is the device bytes of the routed
/// experts alone (the banks a banked load replaces with `slots` members per projection), so
/// `report.device_bytes - routed_bytes` is what a banked load still uploads.
pub(super) struct PackSplit {
    pub(super) report: PackReport,
    pub(super) routed_bytes: u64,
}

pub(super) fn inspect_pack_split(
    source: &StreamedCheckpoint,
    shape: &HostShape,
) -> Result<PackSplit, EngError> {
    let mut walk = Inspector::new(source, Root::Layers);
    let dim = |value: usize| u32::try_from(value).unwrap_or(u32::MAX);
    let h = shape.hidden;
    // The step gathers rows from the table as BF16 (`exl3_global`); another width would load
    // every layer before failing there.
    let embed = walk.plain(EMBED_NAME, &[dim(shape.vocab), dim(h)], true)?;
    if embed != Dtype::Bf16 {
        return Err(refuse(format!(
            "{EMBED_NAME} is {embed:?}, the step needs Bf16"
        )));
    }
    walk.plain(NORM_NAME, &[dim(h)], false)?;
    walk.matrix(HEAD_PREFIX, h, shape.vocab)?;
    let mut routed_total = 0u64;
    for (layer, kind) in shape.kinds.iter().enumerate() {
        walk.plain(
            &walk.root().name(layer, "input_layernorm.weight"),
            &[dim(h)],
            false,
        )?;
        walk.plain(
            &walk.root().name(layer, "post_attention_layernorm.weight"),
            &[dim(h)],
            false,
        )?;
        match kind {
            AttentionKind::Full => walk.attention(layer, shape)?,
            AttentionKind::DeltaNet => walk.delta(layer, shape)?,
            other => return Err(refuse(format!("layer {layer} is {other:?}"))),
        }
        routed_total = routed_total.saturating_add(walk.moe(layer, shape)?);
    }
    let report = walk.report()?;
    Ok(PackSplit {
        report,
        routed_bytes: routed_total,
    })
}

impl Inspector<'_> {
    pub(super) fn attention(&mut self, layer: usize, shape: &HostShape) -> Result<(), EngError> {
        let (h, a) = (shape.hidden, &shape.attn);
        let (q, kv) = (a.heads_q * a.head_dim, a.heads_kv * a.head_dim);
        for (role, k, n) in [
            ("q_proj", h, 2 * q),
            ("k_proj", h, kv),
            ("v_proj", h, kv),
            ("o_proj", q, h),
        ] {
            self.matrix(&self.root().name(layer, &format!("self_attn.{role}")), k, n)?;
        }
        for role in ["q_norm", "k_norm"] {
            let name = self.root().name(layer, &format!("self_attn.{role}.weight"));
            self.plain(&name, &[a.head_dim as u32], false)?;
        }
        Ok(())
    }

    fn delta(&mut self, layer: usize, shape: &HostShape) -> Result<(), EngError> {
        let (h, d) = (shape.hidden, &shape.delta);
        let (z, heads) = (d.value_heads * d.value_dim, d.value_heads as u32);
        for (role, k, n) in [
            ("in_proj_qkv", h, d.conv_dim()),
            ("in_proj_z", h, z),
            ("out_proj", z, h),
        ] {
            self.matrix(
                &self.root().name(layer, &format!("linear_attn.{role}")),
                k,
                n,
            )?;
        }
        for role in ["in_proj_a", "in_proj_b"] {
            let name = self
                .root()
                .name(layer, &format!("linear_attn.{role}.weight"));
            self.plain(&name, &[heads, h as u32], false)?;
        }
        for (role, dims) in [
            ("A_log", vec![heads]),
            ("dt_bias", vec![heads]),
            (
                "conv1d.weight",
                vec![d.conv_dim() as u32, 1, d.conv_kernel as u32],
            ),
            ("norm.weight", vec![d.value_dim as u32]),
        ] {
            self.plain(
                &self.root().name(layer, &format!("linear_attn.{role}")),
                &dims,
                false,
            )?;
        }
        Ok(())
    }

    /// The routed-expert device bytes of the layer (see [`Inspector::moe_block`]).
    pub(super) fn moe(&mut self, layer: usize, shape: &HostShape) -> Result<u64, EngError> {
        let m = &shape.moe;
        self.moe_block(
            layer,
            MoeBlock {
                hidden: shape.hidden,
                experts: m.experts,
                inter: m.inter,
                shared_inter: m.shared_inter,
            },
        )
    }
}

/// The planning summary of one pack directory for a device-free estimate: exactly the
/// loader's own header walk, with its totals copied out for callers outside `family`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Exl3PackPlan {
    /// Device bytes the load will hold ([`PackReport::device_bytes`]: trellises as stored,
    /// scales and plain vectors widened to F32, the embedding as stored).
    pub(crate) device_bytes: u64,
    /// Quantized matrices (a routed expert counts one per projection).
    pub(crate) matrices: usize,
    /// Matrices per rate, keyed by twice the bits per weight.
    pub(crate) rates: std::collections::BTreeMap<u8, usize>,
    /// The codebook marker tensor every matrix of the pack carries, by name.
    pub(crate) codebook: &'static str,
}

/// The name of the codebook marker a matrix carries, for rows that retell the walk.
pub(super) fn codebook_name(marker: Marker) -> &'static str {
    match marker {
        Marker::Mul1 => "Mul1",
        Marker::Mcg => "Mcg",
    }
}

/// Walks the pack of `dir` the way the serving load does (`load_exl3_dir`: pinned model,
/// pinned rotary dimension, no MTP head) and returns its planning facts. A pack of another
/// size, rate, codebook or layout is refused here, exactly as the load would refuse it
/// before the first upload.
pub(crate) fn exl3_plan(dir: &std::path::Path) -> Result<Exl3PackPlan, EngError> {
    let checkpoint = crate::weights::exl3::Checkpoint::open(dir)?;
    let shape = HostShape::from_spec(
        &crate::family::spec::QWEN3_6_35B_A3B,
        super::settings::ROTARY_DIM,
    )?;
    let report = inspect_pack(&checkpoint.source, &shape)?;
    Ok(Exl3PackPlan {
        device_bytes: report.device_bytes,
        matrices: report.matrices,
        rates: report.rates,
        codebook: codebook_name(report.codebook),
    })
}
