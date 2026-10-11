//! Resident Q36 weights, imported once at load (decision 0046 item 6).
//!
//! Non-expert matrices, plain norms, split convolutions, and the F32 router and shared gate become
//! residents with `layers + 2` synchronizations. The 120 expert stacks are read once with positioned
//! reads into page-aligned anonymous pages and wrapped without a copy (decision 0041). They need no
//! synchronization and no page pool, because every expert stays resident for the life of the engine.
//! The embedding table stays in the checkpoint; the token start reads one row per token.

use super::super::settings::{Activation, scale_pairing_error};
use super::layout::{
    ExpertPlan, neg_exp, router_transposed, scalar_gate, split_conv, widen_stack_scales,
    with_f32_scales,
};
use super::ops::{Bytes, Consts, dim, f32_tensor, short_import};
use crate::{
    EngError,
    family::{
        flash_next::stored::{RoleRange, SlicedWeight, StoredWeight, read_stack_into},
        qwen36::{
            host::{DeltaW, HostShape, LayerW, MixerW, MoeW},
            host_load::HostWeights,
        },
    },
    kernels::{HostPages, MlxKernels, PackedPages, PackedWeight, PageRange, ResidentPackedWeight},
    tensor::{Dtype, ExecutionTensor, Tensor},
    weights::StreamedCheckpoint,
};

/// Returns the packed matrix of a stored weight or a `BadRequest`.
fn packed_of(weight: &StoredWeight) -> Result<&PackedWeight, EngError> {
    match weight {
        StoredWeight::Packed(p) => Ok(p),
        StoredWeight::Sliced(_) => Err(EngError::BadRequest("expected a packed matrix".into())),
    }
}

/// Returns the stream description of a sliced stack or table.
fn sliced_of(weight: &StoredWeight) -> Result<SlicedWeight, EngError> {
    match weight {
        StoredWeight::Sliced(s) => Ok(s.clone()),
        StoredWeight::Packed(_) => Err(EngError::BadRequest("expected a sliced weight".into())),
    }
}

/// A packed weight in a batch: as stored, or with its scales and biases widened to F32 (the
/// default of decision 0046, lever Q36-G0).
enum Queued<'w> {
    Stored(&'w PackedWeight),
    Widened(Box<PackedWeight>),
}

impl Queued<'_> {
    fn weight(&self) -> &PackedWeight {
        match self {
            Self::Stored(weight) => weight,
            Self::Widened(weight) => weight.as_ref(),
        }
    }
}

/// Packed weights and plain tensors that import with one synchronization.
///
/// With `f32_scales` every packed weight queues with F32 scales and biases (the default of lever
/// Q36-G0). Otherwise the batch queues the stored weights (BF16 in the Q36 checkpoint).
struct Batch<'w> {
    packed: Vec<Queued<'w>>,
    plain: Vec<Tensor>,
    f32_scales: bool,
}

impl<'w> Batch<'w> {
    fn new(f32_scales: bool) -> Self {
        Self {
            packed: Vec::new(),
            plain: Vec::new(),
            f32_scales,
        }
    }
    fn packed(&mut self, weight: &'w StoredWeight) -> Result<(), EngError> {
        let stored = packed_of(weight)?;
        self.packed.push(if self.f32_scales {
            Queued::Widened(Box::new(with_f32_scales(stored)?))
        } else {
            Queued::Stored(stored)
        });
        Ok(())
    }
    fn plain(&mut self, shape: Vec<u32>, values: &[f32]) -> Result<(), EngError> {
        self.plain.push(f32_tensor(shape, values)?);
        Ok(())
    }
    fn import(self, kernels: &mut MlxKernels) -> Result<Imported, EngError> {
        let plain: Vec<&Tensor> = self.plain.iter().collect();
        let packed: Vec<&PackedWeight> = self.packed.iter().map(Queued::weight).collect();
        let (packed, plain) = kernels.import_packed_many_with(&packed, &plain)?;
        Ok(Imported {
            packed: packed.into_iter(),
            plain: plain.into_iter(),
        })
    }
}

/// Residents of one batch, taken in queue order.
struct Imported {
    packed: std::vec::IntoIter<ResidentPackedWeight>,
    plain: std::vec::IntoIter<ExecutionTensor>,
}

impl Imported {
    fn packed(&mut self) -> Result<ResidentPackedWeight, EngError> {
        self.packed.next().ok_or_else(short_import)
    }
    fn plain(&mut self) -> Result<ExecutionTensor, EngError> {
        self.plain.next().ok_or_else(short_import)
    }
    /// Fails when a queued tensor was not taken: the queue and take orders differ.
    fn finish(mut self) -> Result<(), EngError> {
        if self.packed.next().is_some() || self.plain.next().is_some() {
            return Err(short_import());
        }
        Ok(())
    }
}

/// DeltaNet weights: `conv_taps` `[CD, CK - 1]`, `conv_last` `[CD]`, `neg_exp_a_log` `[VH]` = `-exp(A_log)`,
/// `dt_bias` `[VH]`, and the plain output norm `norm` `[VD]`.
pub(super) struct RDelta {
    pub(super) qkv: ResidentPackedWeight,
    pub(super) z: ResidentPackedWeight,
    pub(super) b: ResidentPackedWeight,
    pub(super) a: ResidentPackedWeight,
    pub(super) out: ResidentPackedWeight,
    pub(super) conv_taps: ExecutionTensor,
    pub(super) conv_last: ExecutionTensor,
    pub(super) neg_exp_a_log: ExecutionTensor,
    pub(super) dt_bias: ExecutionTensor,
    pub(super) norm: ExecutionTensor,
}

impl RDelta {
    fn queue<'w>(
        batch: &mut Batch<'w>,
        w: &DeltaW<'w, StoredWeight>,
        shape: &HostShape,
    ) -> Result<(), EngError> {
        let d = &shape.delta;
        let (taps, last) = split_conv(w.conv, d.conv_dim(), d.conv_kernel)?;
        for weight in [w.qkv, w.z, w.b, w.a, w.out] {
            batch.packed(weight)?;
        }
        batch.plain(vec![dim(d.conv_dim())?, dim(d.conv_kernel - 1)?], &taps)?;
        batch.plain(vec![dim(d.conv_dim())?], &last)?;
        batch.plain(vec![dim(d.value_heads)?], &neg_exp(w.a_log))?;
        batch.plain(vec![dim(d.value_heads)?], w.dt_bias)?;
        batch.plain(vec![dim(d.value_dim)?], w.norm)
    }
    fn take(got: &mut Imported) -> Result<Self, EngError> {
        Ok(Self {
            qkv: got.packed()?,
            z: got.packed()?,
            b: got.packed()?,
            a: got.packed()?,
            out: got.packed()?,
            conv_taps: got.plain()?,
            conv_last: got.plain()?,
            neg_exp_a_log: got.plain()?,
            dt_bias: got.plain()?,
            norm: got.plain()?,
        })
    }
    fn bytes(&self, total: &mut Bytes) -> Result<(), EngError> {
        for w in [&self.qkv, &self.z, &self.b, &self.a, &self.out] {
            total.packed(w)?;
        }
        for t in [
            &self.conv_taps,
            &self.conv_last,
            &self.neg_exp_a_log,
            &self.dt_bias,
            &self.norm,
        ] {
            total.tensor(t)?;
        }
        Ok(())
    }
}

/// Dense attention weights. The norms hold the stored plain weights, `[head_dim]` each.
pub(super) struct RAttn {
    pub(super) q: ResidentPackedWeight,
    pub(super) k: ResidentPackedWeight,
    pub(super) v: ResidentPackedWeight,
    pub(super) o: ResidentPackedWeight,
    pub(super) q_norm: ExecutionTensor,
    pub(super) k_norm: ExecutionTensor,
}

impl RAttn {
    fn queue<'w>(
        batch: &mut Batch<'w>,
        w: &crate::family::qwen36::host::AttnW<'w, StoredWeight>,
    ) -> Result<(), EngError> {
        for weight in [w.q, w.k, w.v, w.o] {
            batch.packed(weight)?;
        }
        batch.plain(vec![dim(w.q_norm.len())?], w.q_norm)?;
        batch.plain(vec![dim(w.k_norm.len())?], w.k_norm)
    }
    fn take(got: &mut Imported) -> Result<Self, EngError> {
        Ok(Self {
            q: got.packed()?,
            k: got.packed()?,
            v: got.packed()?,
            o: got.packed()?,
            q_norm: got.plain()?,
            k_norm: got.plain()?,
        })
    }
    fn bytes(&self, total: &mut Bytes) -> Result<(), EngError> {
        for w in [&self.q, &self.k, &self.v, &self.o] {
            total.packed(w)?;
        }
        total.tensor(&self.q_norm)?;
        total.tensor(&self.k_norm)
    }
}

/// The token mixer of one layer.
pub(super) enum RMixer {
    Delta(RDelta),
    Attn(RAttn),
}

/// The non-expert part of the MoE weights, imported with the layer batch.
struct MoeFront {
    router_t: ExecutionTensor,
    shared_gate: ResidentPackedWeight,
    shared_up: ResidentPackedWeight,
    shared_down: ResidentPackedWeight,
    shared_scalar_t: ExecutionTensor,
}

/// MoE weights. `router_t` is F32 `[H, E]`. `shared_scalar_t` is F32 `[H, 1]`. The three expert
/// stacks have logical shapes `[E, inter, H]`, `[E, inter, H]`, and `[E, H, inter]`.
pub(super) struct RMoe {
    pub(super) router_t: ExecutionTensor,
    pub(super) shared_gate: ResidentPackedWeight,
    pub(super) shared_up: ResidentPackedWeight,
    pub(super) shared_down: ResidentPackedWeight,
    pub(super) shared_scalar_t: ExecutionTensor,
    pub(super) gate: ResidentPackedWeight,
    pub(super) up: ResidentPackedWeight,
    pub(super) down: ResidentPackedWeight,
    /// The pages under the three stacks. Field order is drop order: the pages outlive the stacks.
    pages: HostPages,
}

impl MoeFront {
    fn queue<'w>(
        batch: &mut Batch<'w>,
        w: &MoeW<'w, StoredWeight>,
        shape: &HostShape,
    ) -> Result<(), EngError> {
        let (experts, hidden) = (shape.moe.experts, shape.hidden);
        batch.plain(
            vec![dim(hidden)?, dim(experts)?],
            &router_transposed(packed_of(w.router)?, experts, hidden)?,
        )?;
        for weight in [w.shared_gate, w.shared_up, w.shared_down] {
            batch.packed(weight)?;
        }
        batch.plain(
            vec![dim(hidden)?, 1],
            &scalar_gate(packed_of(w.shared_scalar)?, hidden)?,
        )
    }
    fn take(got: &mut Imported) -> Result<Self, EngError> {
        Ok(Self {
            router_t: got.plain()?,
            shared_gate: got.packed()?,
            shared_up: got.packed()?,
            shared_down: got.packed()?,
            shared_scalar_t: got.plain()?,
        })
    }
}

fn page_range(range: &RoleRange) -> PageRange {
    PageRange {
        offset: range.offset,
        dtype: range.dtype,
        shape: range.shape.clone(),
    }
}

impl RMoe {
    /// Reads all experts of one layer into new pages and wraps the three stacks without a copy.
    fn import_experts(
        kernels: &mut MlxKernels,
        ckpt: &StreamedCheckpoint,
        front: MoeFront,
        stacks: &(SlicedWeight, SlicedWeight, SlicedWeight),
        experts: u32,
        f32_scales: bool,
    ) -> Result<Self, EngError> {
        let sliced = (&stacks.0, &stacks.1, &stacks.2);
        let page = HostPages::page_size();
        // With F32 scales (the default) the plan places each scale and bias role at twice its
        // stored size, and `reads` places the stored BF16 values in the upper half of that role.
        let (plan, reads) = if f32_scales {
            let (plan, reads) = ExpertPlan::new_f32_scales(ckpt, sliced, experts, page)?;
            (plan, Some(reads))
        } else {
            (ExpertPlan::new(ckpt, sliced, experts, page)?, None)
        };
        let mut pages = HostPages::try_new(plan.end)?;
        let ids: Vec<u32> = (0..experts).collect();
        {
            let bytes = pages.try_write().ok_or_else(|| EngError::Backend {
                operation: "expert_pages",
                message: "new expert pages are not writable".into(),
            })?;
            for (index, (stored, layout)) in [&stacks.0, &stacks.1, &stacks.2]
                .into_iter()
                .zip(&plan.stacks)
                .enumerate()
            {
                let read = reads.as_ref().map_or(layout, |reads| &reads[index]);
                read_stack_into(ckpt, stored, &ids, bytes, read)?;
                if reads.is_some() {
                    widen_stack_scales(bytes, layout)?;
                }
            }
        }
        let wrapped: Vec<PackedPages> = plan
            .stacks
            .iter()
            .map(|stack| PackedPages {
                payload: page_range(&stack.payload),
                scales: page_range(&stack.scales),
                biases: stack.biases.as_ref().map(page_range),
                logical_shape: stack.logical_shape.clone(),
                quantization: stack.spec,
            })
            .collect();
        let (imported, plain) = kernels.import_packed_pages(&pages, &wrapped, &[])?;
        if !plain.is_empty() {
            return Err(short_import());
        }
        let [gate, up, down]: [ResidentPackedWeight; 3] =
            imported.try_into().map_err(|_| short_import())?;
        Ok(Self {
            router_t: front.router_t,
            shared_gate: front.shared_gate,
            shared_up: front.shared_up,
            shared_down: front.shared_down,
            shared_scalar_t: front.shared_scalar_t,
            gate,
            up,
            down,
            pages,
        })
    }

    fn bytes(&self, total: &mut Bytes) -> Result<(), EngError> {
        total.tensor(&self.router_t)?;
        total.tensor(&self.shared_scalar_t)?;
        for w in [
            &self.shared_gate,
            &self.shared_up,
            &self.shared_down,
            &self.gate,
            &self.up,
            &self.down,
        ] {
            total.packed(w)?;
        }
        Ok(())
    }

    /// Mapped bytes under the three stacks.
    fn page_bytes(&self) -> Result<u64, EngError> {
        u64::try_from(self.pages.len())
            .map_err(|_| EngError::MemoryPlan("expert page bytes exceed u64".into()))
    }
}

/// Weights of one decoder layer. The two norms are the stored plain weights, `[hidden]` each.
pub(super) struct RLayer {
    pub(super) input_norm: ExecutionTensor,
    pub(super) post_norm: ExecutionTensor,
    pub(super) mixer: RMixer,
    pub(super) moe: RMoe,
}

/// The non-expert part of one layer, before its experts are wrapped.
struct LayerFront {
    input_norm: ExecutionTensor,
    post_norm: ExecutionTensor,
    mixer: RMixer,
    moe: MoeFront,
    stacks: (SlicedWeight, SlicedWeight, SlicedWeight),
}

/// Resident Q36 weights.
pub(super) struct ResidentWeights {
    pub(super) shape: HostShape,
    /// The embedding table, read one row per token from the checkpoint.
    pub(super) embed: SlicedWeight,
    pub(super) lm_head: ResidentPackedWeight,
    pub(super) final_norm: ExecutionTensor,
    pub(super) layers: Vec<RLayer>,
    pub(super) consts: Consts,
}

impl ResidentWeights {
    /// Imports `host` in two phases and consumes it. Every scale and bias is stored as F32
    /// (decision 0046, lever Q36-G0).
    ///
    /// Phase 1 imports every non-expert tensor with `layers + 2` synchronizations: the head, one per
    /// layer, and the constants. The host copies drop before phase 2, so the host copies and the
    /// expert pages never coexist. Phase 2 reads each layer's 256 experts into pages and wraps them
    /// without a copy or a synchronization.
    ///
    /// The checkpoint stores the scales and biases as BF16. MLX casts BF16 scales to the F32 of the
    /// activation on every `qmm` and `gather_qmm`, and an F32 operand skips that cast. BF16 widens
    /// to F32 without loss, so the products do not change. The resident bytes grow by the stored
    /// scale and bias bytes (2,133,114,880 B for the 35B model). The widening of the expert pages
    /// runs in place, so no BF16 copy stays mapped.
    pub(super) fn import(
        kernels: &mut MlxKernels,
        ckpt: &StreamedCheckpoint,
        host: HostWeights,
    ) -> Result<Self, EngError> {
        Self::import_with(kernels, ckpt, host, true)
    }

    /// Like `import`, with the scales and biases kept as stored (BF16). The BF16 half of the one
    /// activation lever of packet Q36-BF16-a: the decoder constructor picks it for BF16
    /// activations, and `ensure_scale_pairing` refuses any other pairing.
    pub(super) fn import_bf16_scales(
        kernels: &mut MlxKernels,
        ckpt: &StreamedCheckpoint,
        host: HostWeights,
    ) -> Result<Self, EngError> {
        Self::import_with(kernels, ckpt, host, false)
    }

    fn import_with(
        kernels: &mut MlxKernels,
        ckpt: &StreamedCheckpoint,
        host: HostWeights,
        f32_scales: bool,
    ) -> Result<Self, EngError> {
        let shape = host.shape().clone();
        let view = host.view();
        let mut batch = Batch::new(f32_scales);
        batch.packed(view.lm_head)?;
        batch.plain(vec![dim(shape.hidden)?], view.final_norm)?;
        let mut got = batch.import(kernels)?;
        let lm_head = got.packed()?;
        let final_norm = got.plain()?;
        got.finish()?;
        let embed = sliced_of(view.embed)?;
        let mut fronts = Vec::with_capacity(view.layers.len());
        for layer in &view.layers {
            fronts.push(import_layer_front(kernels, layer, &shape, f32_scales)?);
        }
        drop(view);
        drop(host);
        let consts = Consts::import(kernels, &shape)?;
        let experts = dim(shape.moe.experts)?;
        let mut layers = Vec::with_capacity(fronts.len());
        for front in fronts {
            let moe =
                RMoe::import_experts(kernels, ckpt, front.moe, &front.stacks, experts, f32_scales)?;
            layers.push(RLayer {
                input_norm: front.input_norm,
                post_norm: front.post_norm,
                mixer: front.mixer,
                moe,
            });
        }
        Ok(Self {
            shape,
            embed,
            lm_head,
            final_norm,
            layers,
            consts,
        })
    }

    /// Converts the layer norms to `dtype` and evaluates them once (packet Q36-BF16-a): the input
    /// and post-attention norm of every layer and the query and key norm of every attention layer
    /// (100 tensors on the real model). One synchronization. The checkpoint stores these weights
    /// in BF16, so the conversion from the F32 copies loses nothing. The final norm, the DeltaNet
    /// weights, the router, and the shared gate stay F32. On error the weights may hold a mix, and
    /// the caller drops the decoder.
    pub(super) fn convert_norms(
        &mut self,
        kernels: &mut MlxKernels,
        dtype: Dtype,
    ) -> Result<(), EngError> {
        let mut slots: Vec<&mut ExecutionTensor> = Vec::new();
        for layer in &mut self.layers {
            slots.push(&mut layer.input_norm);
            slots.push(&mut layer.post_norm);
            if let RMixer::Attn(a) = &mut layer.mixer {
                slots.push(&mut a.q_norm);
                slots.push(&mut a.k_norm);
            }
        }
        let converted = slots
            .iter()
            .map(|slot| kernels.cast(slot, dtype))
            .collect::<Result<Vec<_>, _>>()?;
        let refs: Vec<&ExecutionTensor> = converted.iter().collect();
        kernels.evaluate(&refs)?;
        for (slot, tensor) in slots.into_iter().zip(converted) {
            *slot = tensor;
        }
        Ok(())
    }

    /// Checks the scale pairing of packet Q36-BF16-a: every scale and bias of every packed weight
    /// must have the dtype that matches `activation` (862 parts over 431 packed weights on the
    /// real model). `settings::scale_pairing_error` owns the rule; a mismatched pair returns
    /// `BadRequest` before the constructor changes anything else.
    pub(super) fn ensure_scale_pairing(&self, activation: Activation) -> Result<(), EngError> {
        let census = self.scale_census();
        match scale_pairing_error(
            census.weights,
            census.f32_parts,
            census.bf16_parts,
            census.other_parts,
            activation,
        ) {
            Some(message) => Err(EngError::BadRequest(message)),
            None => Ok(()),
        }
    }

    /// Resident bytes of all weights and constants, from the tensors' own sizes. The expert stacks
    /// are included once.
    pub(super) fn byte_len(&self) -> Result<u64, EngError> {
        let mut total = Bytes::default();
        total.packed(&self.lm_head)?;
        total.tensor(&self.final_norm)?;
        for layer in &self.layers {
            total.tensor(&layer.input_norm)?;
            total.tensor(&layer.post_norm)?;
            match &layer.mixer {
                RMixer::Delta(d) => d.bytes(&mut total)?,
                RMixer::Attn(a) => a.bytes(&mut total)?,
            }
            layer.moe.bytes(&mut total)?;
        }
        self.consts.bytes(&mut total)?;
        Ok(total.total())
    }

    /// Mapped bytes of the expert pages of all layers.
    pub(super) fn expert_page_bytes(&self) -> Result<u64, EngError> {
        let mut total = Bytes::default();
        for layer in &self.layers {
            total.add(layer.moe.page_bytes()?)?;
        }
        Ok(total.total())
    }

    /// Counts the packed weights and the dtypes of their scale and bias tensors.
    pub(super) fn scale_census(&self) -> ScaleCensus {
        let mut census = ScaleCensus::default();
        let mut visit = |weight: &ResidentPackedWeight| {
            census.weights += 1;
            for part in [Some(weight.scales()), weight.biases()]
                .into_iter()
                .flatten()
            {
                match part.dtype() {
                    Dtype::F32 => census.f32_parts += 1,
                    Dtype::Bf16 => census.bf16_parts += 1,
                    _ => census.other_parts += 1,
                }
            }
        };
        visit(&self.lm_head);
        for layer in &self.layers {
            match &layer.mixer {
                RMixer::Delta(d) => [&d.qkv, &d.z, &d.b, &d.a, &d.out]
                    .into_iter()
                    .for_each(&mut visit),
                RMixer::Attn(a) => [&a.q, &a.k, &a.v, &a.o].into_iter().for_each(&mut visit),
            }
            let moe = &layer.moe;
            [
                &moe.shared_gate,
                &moe.shared_up,
                &moe.shared_down,
                &moe.gate,
                &moe.up,
                &moe.down,
            ]
            .into_iter()
            .for_each(&mut visit);
        }
        census
    }
}

/// The packed weights of a model and the dtypes of their scale and bias tensors.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct ScaleCensus {
    /// Packed weights, expert stacks included.
    pub(super) weights: u64,
    pub(super) f32_parts: u64,
    pub(super) bf16_parts: u64,
    pub(super) other_parts: u64,
}

/// Imports the non-expert tensors of one layer with one synchronization.
fn import_layer_front(
    kernels: &mut MlxKernels,
    layer: &LayerW<'_, StoredWeight>,
    shape: &HostShape,
    f32_scales: bool,
) -> Result<LayerFront, EngError> {
    let mut batch = Batch::new(f32_scales);
    batch.plain(vec![dim(shape.hidden)?], layer.input_norm)?;
    batch.plain(vec![dim(shape.hidden)?], layer.post_norm)?;
    match &layer.mixer {
        MixerW::Delta(w) => RDelta::queue(&mut batch, w, shape)?,
        MixerW::Attn(w) => RAttn::queue(&mut batch, w)?,
    }
    MoeFront::queue(&mut batch, &layer.moe, shape)?;
    let mut got = batch.import(kernels)?;
    let input_norm = got.plain()?;
    let post_norm = got.plain()?;
    let mixer = match &layer.mixer {
        MixerW::Delta(_) => RMixer::Delta(RDelta::take(&mut got)?),
        MixerW::Attn(_) => RMixer::Attn(RAttn::take(&mut got)?),
    };
    let moe = MoeFront::take(&mut got)?;
    got.finish()?;
    Ok(LayerFront {
        input_norm,
        post_norm,
        mixer,
        moe,
        stacks: (
            sliced_of(layer.moe.gate)?,
            sliced_of(layer.moe.up)?,
            sliced_of(layer.moe.down)?,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::super::ops::{assert_parity, u32_tensor, values};
    use super::*;
    use crate::family::project::Projector;
    use crate::family::qwen36::fixtures::Entry;
    use crate::family::qwen36::host_cpu::CpuProjector;
    use crate::family::qwen36::tiny::{self, TINY, TINY_GQA};
    use crate::family::spec::FamilySpec;
    use crate::kernels::QuantizationMode;

    /// A checkpoint of `spec` and a fresh host load of it.
    fn load(spec: &FamilySpec) -> (StreamedCheckpoint, HostWeights) {
        let ckpt = tiny::checkpoint(spec);
        let host = HostWeights::load(&ckpt, spec, &tiny::config(spec)).unwrap();
        (ckpt, host)
    }

    fn shape_of(weight: &ResidentPackedWeight) -> Vec<u32> {
        weight.logical_shape().to_vec()
    }

    #[test]
    fn import_keeps_the_shape_of_every_role() {
        // Tiny GQA model: hidden 64, 4 experts, inter 64, conv dim 128, kernel 4, 2 value heads of 32.
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load(&TINY_GQA);
        let (_, oracle) = load(&TINY_GQA);
        let mut kernels = MlxKernels::try_new().unwrap();
        let w = ResidentWeights::import(&mut kernels, &ckpt, host).unwrap();
        assert_eq!(w.layers.len(), 4);
        assert_eq!(shape_of(&w.lm_head), [8, 64]);
        assert_eq!(w.final_norm.shape(), [64]);
        // The embedding keeps the stream description of a fresh host load of the same checkpoint.
        let embed = sliced_of(oracle.view().embed).unwrap();
        assert_eq!(w.embed, embed);
        let RMixer::Delta(d) = &w.layers[0].mixer else {
            panic!("layer 0 is DeltaNet");
        };
        assert_eq!(shape_of(&d.qkv), [128, 64]);
        assert_eq!(shape_of(&d.z), [64, 64]);
        assert_eq!(shape_of(&d.b), [2, 64]);
        assert_eq!(shape_of(&d.out), [64, 64]);
        assert_eq!(d.conv_taps.shape(), [128, 3]);
        assert_eq!(d.conv_last.shape(), [128]);
        assert_eq!(d.neg_exp_a_log.shape(), [2]);
        assert_eq!(d.dt_bias.shape(), [2]);
        assert_eq!(d.norm.shape(), [32]);
        let RMixer::Attn(a) = &w.layers[1].mixer else {
            panic!("layer 1 is attention");
        };
        // 4 query heads of 32 with a gate: 4 * 64 = 256 rows. 2 key-value heads: 64 rows.
        assert_eq!(shape_of(&a.q), [256, 64]);
        assert_eq!(shape_of(&a.k), [64, 64]);
        assert_eq!(shape_of(&a.v), [64, 64]);
        assert_eq!(shape_of(&a.o), [64, 128]);
        assert_eq!((a.q_norm.shape(), a.k_norm.shape()), (&[32][..], &[32][..]));
        let moe = &w.layers[3].moe;
        assert_eq!(moe.router_t.shape(), [64, 4]);
        assert_eq!(moe.shared_scalar_t.shape(), [64, 1]);
        assert_eq!(shape_of(&moe.shared_gate), [64, 64]);
        assert_eq!(shape_of(&moe.gate), [4, 64, 64]);
        assert_eq!(shape_of(&moe.up), [4, 64, 64]);
        assert_eq!(shape_of(&moe.down), [4, 64, 64]);
    }

    /// The bytes that the import must make resident, summed from the stored tensor list by rule:
    /// packed payloads stay as stored; scales and biases take 4 bytes per element when they are
    /// F32 (the default) and their stored bytes when they stay BF16; dense BF16 tensors become F32;
    /// the embedding stays in the checkpoint; the router and the shared gate become F32 matrices;
    /// the three constants take 4 bytes each.
    fn expected_bytes(entries: &[Entry], layers: u64, f32_scales: bool) -> u64 {
        let mut total = 0u64;
        for (name, dtype, shape, bytes) in entries {
            let elements: u64 = shape.iter().map(|&d| u64::from(d)).product();
            let payload = *dtype == "U32";
            let scale_part = name.ends_with(".scales") || name.ends_with(".biases");
            let converted = name.contains("mlp.gate.") || name.contains("shared_expert_gate.");
            if name.contains("embed_tokens") || converted {
                continue;
            }
            total += if payload || (scale_part && !f32_scales) {
                bytes.len() as u64
            } else {
                elements * 4
            };
        }
        // Per layer: router [64, 4] and shared gate [64, 1] as F32.
        total + layers * (64 * 4 * 4 + 64 * 4) + 12
    }

    /// The mapped bytes of the expert pages of the tiny GQA model: 4 layers * 3 stacks, each with
    /// a payload of 4 * 64 * 8 words * 4 B = 8,192 B rounded to a page, then the scales and the
    /// biases of 4 experts * 256 B stored (BF16) or 4 * 512 B (F32), each rounded to a page.
    fn expected_page_bytes(f32_scales: bool) -> u64 {
        let page = HostPages::page_size() as u64;
        let role = |bytes: u64| bytes.div_ceil(page) * page;
        let scale_role = if f32_scales { 2048 } else { 1024 };
        4 * 3 * (role(8192) + 2 * role(scale_role))
    }

    #[test]
    fn the_resident_byte_total_follows_the_stored_tensor_list() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load(&TINY_GQA);
        let mut kernels = MlxKernels::try_new().unwrap();
        let w = ResidentWeights::import(&mut kernels, &ckpt, host).unwrap();
        assert_eq!(
            w.byte_len().unwrap(),
            expected_bytes(&tiny::entries(&TINY_GQA), 4, true)
        );
        assert_eq!(w.expert_page_bytes().unwrap(), expected_page_bytes(true));
    }

    /// The scale and bias census that the import must produce: the weights by rule, every part of
    /// the stated dtype.
    fn assert_scale_census(w: &ResidentWeights, f32_scales: bool) {
        // The head, five DeltaNet or four attention weights per layer, and six MoE weights.
        let mixers: u64 = w
            .layers
            .iter()
            .map(|layer| match layer.mixer {
                RMixer::Delta(_) => 5,
                RMixer::Attn(_) => 4,
            })
            .sum();
        let weights = 1 + mixers + 6 * w.layers.len() as u64;
        let parts = 2 * weights;
        let (want_f32, want_bf16) = if f32_scales { (parts, 0) } else { (0, parts) };
        let census = w.scale_census();
        assert_eq!(
            (
                census.weights,
                census.f32_parts,
                census.bf16_parts,
                census.other_parts
            ),
            (weights, want_f32, want_bf16, 0)
        );
    }

    #[test]
    fn import_stores_every_scale_and_bias_as_f32() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load(&TINY_GQA);
        let mut kernels = MlxKernels::try_new().unwrap();
        let w = ResidentWeights::import(&mut kernels, &ckpt, host).unwrap();
        assert_scale_census(&w, true);
    }

    #[test]
    fn the_bf16_comparison_import_keeps_the_stored_scales_and_biases() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load(&TINY_GQA);
        let mut kernels = MlxKernels::try_new().unwrap();
        let w = ResidentWeights::import_bf16_scales(&mut kernels, &ckpt, host).unwrap();
        assert_scale_census(&w, false);
        assert_eq!(
            w.byte_len().unwrap(),
            expected_bytes(&tiny::entries(&TINY_GQA), 4, false)
        );
        assert_eq!(w.expert_page_bytes().unwrap(), expected_page_bytes(false));
    }

    #[test]
    fn dense_tensors_are_loaded_as_stored_and_the_delta_constants_are_derived_by_rule() {
        // Norm weights apply as stored (Flash-Next adds 1). The convolution splits into its first
        // three taps per channel and its last column. `neg_exp_a_log` is -exp(A_log). The expected
        // values come from the stored tensors, read through the checkpoint reader.
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load(&TINY);
        let mut kernels = MlxKernels::try_new().unwrap();
        let w = ResidentWeights::import(&mut kernels, &ckpt, host).unwrap();
        let stored = |key: &str, len: usize| {
            crate::family::flash_next::stored::host_f32(&ckpt, key, len).unwrap()
        };
        let p = "language_model.model.layers.";
        assert_eq!(
            values(&mut kernels, &w.final_norm).unwrap(),
            stored("language_model.model.norm.weight", 64),
            "final norm"
        );
        assert_eq!(
            values(&mut kernels, &w.layers[1].input_norm).unwrap(),
            stored(&format!("{p}1.input_layernorm.weight"), 64),
            "layer 1 input norm"
        );
        assert_eq!(
            values(&mut kernels, &w.layers[0].post_norm).unwrap(),
            stored(&format!("{p}0.post_attention_layernorm.weight"), 64),
            "layer 0 post norm"
        );
        let RMixer::Delta(d) = &w.layers[0].mixer else {
            panic!("layer 0 is DeltaNet");
        };
        let conv = stored(&format!("{p}0.linear_attn.conv1d.weight"), 128 * 4);
        let taps: Vec<f32> = (0..128)
            .flat_map(|c| conv[c * 4..c * 4 + 3].to_vec())
            .collect();
        let last: Vec<f32> = (0..128).map(|c| conv[c * 4 + 3]).collect();
        assert_eq!(
            values(&mut kernels, &d.conv_taps).unwrap(),
            taps,
            "conv taps"
        );
        assert_eq!(
            values(&mut kernels, &d.conv_last).unwrap(),
            last,
            "conv last"
        );
        let a_log = stored(&format!("{p}0.linear_attn.A_log"), 2);
        let got = values(&mut kernels, &d.neg_exp_a_log).unwrap();
        for (g, a) in got.iter().zip(&a_log) {
            assert!(
                (g + f64::from(*a).exp() as f32).abs() <= 1e-6,
                "{g} against A_log {a}"
            );
        }
        assert_eq!(
            values(&mut kernels, &d.dt_bias).unwrap(),
            stored(&format!("{p}0.linear_attn.dt_bias"), 2),
            "dt bias"
        );
        assert_eq!(
            values(&mut kernels, &d.norm).unwrap(),
            stored(&format!("{p}0.linear_attn.norm.weight"), 32),
            "delta norm"
        );
        let RMixer::Attn(a) = &w.layers[1].mixer else {
            panic!("layer 1 is attention");
        };
        assert_eq!(
            values(&mut kernels, &a.q_norm).unwrap(),
            stored(&format!("{p}1.self_attn.q_norm.weight"), 32),
            "q norm"
        );
        assert_eq!(
            values(&mut kernels, &a.k_norm).unwrap(),
            stored(&format!("{p}1.self_attn.k_norm.weight"), 32),
            "k norm"
        );
    }

    #[test]
    fn router_and_shared_gate_equal_the_cpu_projection_of_unit_vectors() {
        // Projecting the unit vector e_h with the CPU projector returns column h of the packed
        // matrix, so router_t[h][e] must equal projection(e_h)[e].
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load(&TINY);
        let (_, oracle) = load(&TINY);
        let mut kernels = MlxKernels::try_new().unwrap();
        let w = ResidentWeights::import(&mut kernels, &ckpt, host).unwrap();
        let view = oracle.view();
        let mut cpu = CpuProjector::new(&ckpt, 1);
        let router = values(&mut kernels, &w.layers[1].moe.router_t).unwrap();
        let scalar = values(&mut kernels, &w.layers[1].moe.shared_scalar_t).unwrap();
        for h in [0usize, 1, 17, 63] {
            let mut unit = vec![0.0f32; 64];
            unit[h] = 1.0;
            let column = cpu.linear(view.layers[1].moe.router, &unit).unwrap();
            assert_eq!(
                &router[h * 4..h * 4 + 4],
                column.as_slice(),
                "router column {h}"
            );
            let gate = cpu.linear(view.layers[1].moe.shared_scalar, &unit).unwrap();
            assert_eq!(scalar[h], gate[0], "shared gate element {h}");
        }
    }

    #[test]
    fn every_expert_of_a_stack_matches_the_cpu_projection() {
        // gather_qmm with one index e selects expert e of the wrapped stack. The CPU projector reads
        // the same expert from the checkpoint and multiplies it itself.
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load(&TINY);
        let (_, oracle) = load(&TINY);
        let mut kernels = MlxKernels::try_new().unwrap();
        let w = ResidentWeights::import(&mut kernels, &ckpt, host).unwrap();
        let view = oracle.view();
        let mut cpu = CpuProjector::new(&ckpt, 1);
        let x: Vec<f32> = (0..64).map(|i| ((i % 7) as f32 - 3.0) * 0.125).collect();
        let x3 = kernels
            .import_tensor(&f32_tensor(vec![1, 1, 64], &x).unwrap())
            .unwrap();
        for (name, stack, host_stack) in [
            ("gate", &w.layers[0].moe.gate, view.layers[0].moe.gate),
            ("up", &w.layers[0].moe.up, view.layers[0].moe.up),
            ("down", &w.layers[1].moe.down, view.layers[1].moe.down),
        ] {
            for e in 0..4u32 {
                let ids = kernels
                    .import_tensor(&u32_tensor(vec![1], &[e]).unwrap())
                    .unwrap();
                let got = kernels.gather_qmm(&x3, stack, &ids).unwrap();
                let got = values(&mut kernels, &got).unwrap();
                let want = cpu.expert_linear(host_stack, e, &x).unwrap();
                assert_parity(&got, &want, &format!("{name} expert {e}"));
            }
        }
    }

    /// Projects a dense pattern and a unit vector through the resident weight and through the CPU
    /// reference of the stored weight. Equal outputs pin the role, the row order, and the packing.
    fn assert_role(
        kernels: &mut MlxKernels,
        cpu: &mut CpuProjector<'_>,
        resident: &ResidentPackedWeight,
        host: &StoredWeight,
        role: &str,
    ) {
        let input = resident.logical_shape()[1];
        let dense: Vec<f32> = (0..input).map(|i| ((i % 7) as f32 - 3.0) * 0.125).collect();
        let mut unit = vec![0.0f32; input as usize];
        unit[input as usize - 1] = 1.0;
        for (label, x) in [("dense", dense), ("unit", unit)] {
            let tensor = kernels
                .import_tensor(&f32_tensor(vec![1, input], &x).unwrap())
                .unwrap();
            let got = kernels.qmm(&tensor, resident).unwrap();
            let got = values(kernels, &got).unwrap();
            let want = cpu.linear(host, &x).unwrap();
            assert_parity(&got, &want, &format!("{role} {label}"));
        }
        // The stored quantization stays: 4-bit affine with groups of 32 in the tiny models.
        let spec = resident.quantization();
        assert_eq!(spec, packed_of(host).unwrap().quantization(), "{role}");
        assert_eq!(
            (spec.mode, spec.bits, spec.group_size),
            (QuantizationMode::Affine, 4, 32),
            "{role} quantization"
        );
    }

    #[test]
    fn every_packed_role_matches_the_cpu_projection_and_keeps_its_quantization() {
        // Same-shape roles (b and a, z and out, shared gate and up, k and v) hold different values in
        // the tiny model, so a swapped role fails the comparison with the stored weight.
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load(&TINY_GQA);
        let (_, oracle) = load(&TINY_GQA);
        let mut kernels = MlxKernels::try_new().unwrap();
        let w = ResidentWeights::import(&mut kernels, &ckpt, host).unwrap();
        let view = oracle.view();
        let mut cpu = CpuProjector::new(&ckpt, 1);
        assert_role(&mut kernels, &mut cpu, &w.lm_head, view.lm_head, "lm_head");
        let RMixer::Delta(d) = &w.layers[0].mixer else {
            panic!("layer 0 is DeltaNet");
        };
        let MixerW::Delta(hd) = &view.layers[0].mixer else {
            panic!("host layer 0 is DeltaNet");
        };
        for (role, resident, stored) in [
            ("delta qkv", &d.qkv, hd.qkv),
            ("delta z", &d.z, hd.z),
            ("delta b", &d.b, hd.b),
            ("delta a", &d.a, hd.a),
            ("delta out", &d.out, hd.out),
        ] {
            assert_role(&mut kernels, &mut cpu, resident, stored, role);
        }
        let RMixer::Attn(a) = &w.layers[1].mixer else {
            panic!("layer 1 is attention");
        };
        let MixerW::Attn(ha) = &view.layers[1].mixer else {
            panic!("host layer 1 is attention");
        };
        for (role, resident, stored) in [
            ("attention q", &a.q, ha.q),
            ("attention k", &a.k, ha.k),
            ("attention v", &a.v, ha.v),
            ("attention o", &a.o, ha.o),
        ] {
            assert_role(&mut kernels, &mut cpu, resident, stored, role);
        }
        for (index, layer) in w.layers.iter().enumerate() {
            let host_moe = &view.layers[index].moe;
            for (role, resident, stored) in [
                ("shared gate", &layer.moe.shared_gate, host_moe.shared_gate),
                ("shared up", &layer.moe.shared_up, host_moe.shared_up),
                ("shared down", &layer.moe.shared_down, host_moe.shared_down),
            ] {
                let role = format!("layer {index} {role}");
                assert_role(&mut kernels, &mut cpu, resident, stored, &role);
            }
            for stack in [&layer.moe.gate, &layer.moe.up, &layer.moe.down] {
                let spec = stack.quantization();
                assert_eq!(
                    (spec.mode, spec.bits, spec.group_size),
                    (QuantizationMode::Affine, 4, 32),
                    "layer {index} expert stack quantization"
                );
            }
        }
    }

    #[test]
    fn a_checkpoint_without_the_layers_fails_with_a_checkpoint_error() {
        // The host load comes from the four-layer model. The reader holds only two layers, so the
        // expert pages of layer 2 cannot be placed.
        let _serial = crate::native_test_lock();
        let (_, host) = load(&TINY_GQA);
        let short = tiny::checkpoint(&TINY);
        let mut kernels = MlxKernels::try_new().unwrap();
        let err = ResidentWeights::import(&mut kernels, &short, host).err();
        assert!(matches!(err, Some(EngError::Checkpoint(_))), "{err:?}");
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn import_synchronizes_layers_plus_two_times() {
        // The page imports need no synchronization. The zero-copy count is checked on the real
        // checkpoint only: the tiny stacks are not whole multiples of the 16,384-byte page.
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load(&TINY_GQA);
        let mut kernels = MlxKernels::try_new().unwrap();
        let syncs = kernels.qualification_sync_count();
        let _w = ResidentWeights::import(&mut kernels, &ckpt, host).unwrap();
        // The head, four layers, and the constants.
        assert_eq!(kernels.qualification_sync_count() - syncs, 6);
    }

    /// The pinned checkpoint, loaded the way the server loads it. Run on the Mac with
    /// `Q36_CKPT_DIR=/Users/dan/edge0/models/edge0-35b` and `--ignored --nocapture`. Optional:
    /// `Q36_OUT_DIR` (a path that does not exist yet) receives `load.json`.
    #[test]
    #[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
    fn real_checkpoint_loads_within_the_memory_plan() {
        use super::super::real::{env_path, open, rss_kb};
        use crate::family::spec::QWEN3_6_35B_A3B;
        let _serial = crate::native_test_lock();
        let dir = env_path("Q36_CKPT_DIR");
        let real = open(&dir);
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.reset_peak_memory().unwrap();
        let rss_before = rss_kb();
        let started = std::time::Instant::now();
        let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
        assert_eq!(
            host.keys().len(),
            1757,
            "the load reads every tensor of the index"
        );
        let host_seconds = started.elapsed().as_secs_f64();
        let rss_host = rss_kb();
        #[cfg(feature = "native-fault-qualification")]
        let (syncs, copies) = (
            kernels.qualification_sync_count(),
            kernels.qualification_page_copies(),
        );
        let f32_scales = super::super::scale_knob::scales_f32_requested();
        let started = std::time::Instant::now();
        #[cfg(feature = "native-fault-qualification")]
        let w = if f32_scales {
            ResidentWeights::import(&mut kernels, &real.ckpt, host)
        } else {
            ResidentWeights::import_bf16_scales(&mut kernels, &real.ckpt, host)
        }
        .unwrap();
        #[cfg(not(feature = "native-fault-qualification"))]
        let w = ResidentWeights::import(&mut kernels, &real.ckpt, host).unwrap();
        let import_seconds = started.elapsed().as_secs_f64();
        let stats = kernels.memory_stats().unwrap();
        let rss_after = rss_kb();
        // Sums from the pinned shard headers (decision 0046 item 6). With BF16 scales and biases
        // (the comparison run): expert stacks 18,119,393,280 B; every resident tensor
        // 19,286,886,924 B (stored 19,508,787,456 B, minus the embedding 286,064,640 B and the
        // int8 router and shared gate, plus F32 dense tensors and routers). The default F32
        // storage doubles the BF16 scale and bias bytes of the packed weights that stay packed.
        // The pinned headers hold 2,013,265,920 B of them in the expert stacks and 119,848,960 B
        // in the 311 dense packed weights (the embedding, the router, and the shared expert gate
        // excluded), 2,133,114,880 B in all. That gives 20,132,659,200 B of expert pages and
        // 21,420,001,804 B resident.
        assert_eq!(w.layers.len(), 40);
        let (want_pages, want_bytes) = if f32_scales {
            (20_132_659_200, 21_420_001_804)
        } else {
            (18_119_393_280, 19_286_886_924)
        };
        assert_eq!(w.expert_page_bytes().unwrap(), want_pages);
        assert_eq!(w.byte_len().unwrap(), want_bytes);
        let census = w.scale_census();
        let (want_f32, want_bf16) = if f32_scales { (862, 0) } else { (0, 862) };
        assert_eq!(
            (
                census.weights,
                census.f32_parts,
                census.bf16_parts,
                census.other_parts
            ),
            (431, want_f32, want_bf16, 0),
            "scale and bias dtypes"
        );
        // The load bound is the resident total plus the allowance of the BF16 bound: 19.5 GiB
        // (20,937,965,568 B) over 19,286,886,924 B is 1,651,078,644 B. With F32 scales that is
        // 21,420,001,804 + 1,651,078,644 = 23,071,080,448 B, rounded up to 21.5 GiB
        // (23,085,449,216 B). The BF16 comparison run keeps 19.5 GiB.
        let bound: u64 = if f32_scales {
            23_085_449_216
        } else {
            20_937_965_568
        };
        let within_plan = stats.peak <= bound;
        assert!(
            within_plan,
            "MLX peak {} B exceeds the load bound {bound} B",
            stats.peak
        );
        let report = serde_json::json!({
            "host_load_seconds": host_seconds, "import_seconds": import_seconds,
            "mlx_active": stats.active, "mlx_cache": stats.cache, "mlx_peak": stats.peak,
            "rss_kb_before": rss_before, "rss_kb_after_host_load": rss_host, "rss_kb_after_import": rss_after,
            "resident_bytes": w.byte_len().unwrap(), "expert_page_bytes": w.expert_page_bytes().unwrap(),
            "scale_dtype": if f32_scales { "f32" } else { "bf16" },
            "scale_parts_f32": census.f32_parts, "scale_parts_bf16": census.bf16_parts,
            "mlx_load_bound_bytes": bound, "mlx_peak_within_load_bound": within_plan,
        });
        #[cfg(feature = "native-fault-qualification")]
        let report = {
            let (syncs_after, copies_after) = (
                kernels.qualification_sync_count(),
                kernels.qualification_page_copies(),
            );
            assert_eq!(syncs_after - syncs, 42, "layers + 2 synchronizations");
            assert_eq!(
                copies_after - copies,
                0,
                "no expert page import used the copy path"
            );
            let mut report = report;
            report["syncs"] = (syncs_after - syncs).into();
            report["page_copies"] = (copies_after - copies).into();
            report
        };
        println!("{report}");
        if let Ok(out) = std::env::var("Q36_OUT_DIR") {
            std::fs::create_dir(&out).expect("Q36_OUT_DIR must not exist yet");
            std::fs::write(
                std::path::Path::new(&out).join("load.json"),
                serde_json::to_vec_pretty(&report).unwrap(),
            )
            .unwrap();
        }
        // Packet Q36-BF16-a, the 337,920 B reconciliation: the totals above count the norms as
        // imported (F32). The product BF16 constructor converts the 100 norm tensors (40 input
        // and 40 post norms of 2048, 10 query and 10 key norms of 256: 168,960 elements) to BF16,
        // which saves 337,920 B: 19,286,886,924 - 337,920 = 19,286,549,004 B, the `weights_bytes`
        // that the BF16 tool-forward log reported. One extra synchronization converts them.
        if !f32_scales {
            // The product constructor turns the BF16 convert on before `convert_norms`
            // (`ResidentDecoder::new_paired`); without it `cast` refuses the BF16 output with
            // `convert: output dtype must be F32 or an allowed BF16`.
            kernels.allow_bf16(true);
            let mut w = w;
            w.convert_norms(&mut kernels, crate::tensor::Dtype::Bf16)
                .unwrap();
            assert_eq!(
                w.byte_len().unwrap(),
                19_286_549_004,
                "the product BF16 decoder counts the converted norms"
            );
        }
    }
}
