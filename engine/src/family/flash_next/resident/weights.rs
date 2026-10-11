//! Resident Flash-Next weights, imported once at load (decision 0039; S1-C plan section 1).
//!
//! Non-expert matrices, `1 + w` norms, split convolutions, and F32 routers transposed to `[H, E]`
//! become residents. Expert stacks, the embedding, and the PLE tables stay streamed (`SlicedWeight`).

use super::super::{
    deltanet::DeltaWeights,
    hyper::HyperWeights,
    load::FlashWeights,
    moe::MoeWeights,
    ple::PleWeights,
    qsa::QsaWeights,
    step::{AttnWeights, ModelShape},
    stored::{SlicedWeight, StoredWeight},
};
use super::ops::{Bytes, Consts, dim, f32_tensor, short_import};
use crate::{
    EngError,
    kernels::{MlxKernels, PackedWeight, ResidentPackedWeight},
    tensor::{ExecutionTensor, Tensor},
};

/// Returns `1 + w` for a zero-centered norm weight.
fn one_plus(weight: &[f32]) -> Vec<f32> {
    weight.iter().map(|w| 1.0 + w).collect()
}

/// Splits a row-major `[rows, kernel]` convolution into its first `kernel - 1` taps and its last column.
fn split_conv(conv: &[f32], rows: usize, kernel: usize) -> Result<(Vec<f32>, Vec<f32>), EngError> {
    if kernel < 2 || Some(conv.len()) != rows.checked_mul(kernel) {
        return Err(EngError::BadRequest(
            "convolution length does not match its shape".into(),
        ));
    }
    let (mut taps, mut last) = (
        Vec::with_capacity(conv.len() - rows),
        Vec::with_capacity(rows),
    );
    for row in conv.chunks_exact(kernel) {
        if let Some((tail, head)) = row.split_last() {
            taps.extend_from_slice(head);
            last.push(*tail);
        }
    }
    Ok((taps, last))
}

/// Transposes a row-major `[rows, cols]` matrix to `[cols, rows]`.
fn transpose(values: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, EngError> {
    let mismatch = || EngError::BadRequest("matrix length does not match its shape".into());
    if Some(values.len()) != rows.checked_mul(cols) {
        return Err(mismatch());
    }
    (0..cols)
        .flat_map(|c| (0..rows).map(move |r| r * cols + c))
        .map(|i| values.get(i).copied())
        .collect::<Option<Vec<f32>>>()
        .ok_or_else(mismatch)
}

/// Returns the stream description of a sliced stack or table.
fn sliced(weight: &StoredWeight) -> Result<SlicedWeight, EngError> {
    match weight {
        StoredWeight::Sliced(s) => Ok(s.clone()),
        StoredWeight::Packed(_) => Err(EngError::BadRequest("expected a sliced weight".into())),
    }
}

/// Packed weights and plain tensors that import with one synchronization.
#[derive(Default)]
struct Batch<'w> {
    packed: Vec<&'w PackedWeight>,
    plain: Vec<Tensor>,
}

impl<'w> Batch<'w> {
    fn packed(&mut self, weight: &'w StoredWeight) -> Result<(), EngError> {
        match weight {
            StoredWeight::Packed(p) => {
                self.packed.push(p);
                Ok(())
            }
            StoredWeight::Sliced(_) => Err(EngError::BadRequest("expected a packed matrix".into())),
        }
    }
    fn plain(&mut self, shape: Vec<u32>, values: &[f32]) -> Result<(), EngError> {
        self.plain.push(f32_tensor(shape, values)?);
        Ok(())
    }
    fn import(self, kernels: &mut MlxKernels) -> Result<Imported, EngError> {
        let plain: Vec<&Tensor> = self.plain.iter().collect();
        let (packed, plain) = kernels.import_packed_many_with(&self.packed, &plain)?;
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

/// One hyper-connection. `norm1p` `[C, H]` holds `1 + w`. The root mixer has no `inject`.
pub(super) struct RHyper {
    pub(super) norm1p: ExecutionTensor,
    pub(super) down: ResidentPackedWeight,
    pub(super) up: ResidentPackedWeight,
    pub(super) inject: Option<ResidentPackedWeight>,
}

impl RHyper {
    fn queue<'w>(
        batch: &mut Batch<'w>,
        w: &HyperWeights<'w, StoredWeight>,
        shape: &ModelShape,
    ) -> Result<(), EngError> {
        batch.plain(
            vec![dim(shape.hyper.branches())?, dim(shape.hyper.hidden())?],
            &one_plus(w.norm),
        )?;
        batch.packed(w.down)?;
        batch.packed(w.up)?;
        if let Some(inject) = w.inject {
            batch.packed(inject)?;
        }
        Ok(())
    }
    fn take(got: &mut Imported, inject: bool) -> Result<Self, EngError> {
        Ok(Self {
            norm1p: got.plain()?,
            down: got.packed()?,
            up: got.packed()?,
            inject: if inject { Some(got.packed()?) } else { None },
        })
    }
    fn bytes(&self, total: &mut Bytes) -> Result<(), EngError> {
        total.tensor(&self.norm1p)?;
        total.packed(&self.down)?;
        total.packed(&self.up)?;
        if let Some(inject) = &self.inject {
            total.packed(inject)?;
        }
        Ok(())
    }
}

/// DeltaNet weights: `conv_taps` `[CD, CK - 1]`, `conv_last` `[CD]`, `neg_exp_a_log` `[VH]` = `-exp(A_log)`, `dt_bias` `[VH]`, plain `norm` `[VD]`.
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
        w: &DeltaWeights<'w, StoredWeight>,
        shape: &ModelShape,
    ) -> Result<(), EngError> {
        let d = &shape.delta;
        let (taps, last) = split_conv(w.conv, d.conv_dim(), d.conv_kernel())?;
        for weight in [w.qkv, w.z, w.b, w.a, w.out] {
            batch.packed(weight)?;
        }
        let neg: Vec<f32> = w.a_log.iter().map(|v| -(v.exp())).collect();
        batch.plain(vec![dim(d.conv_dim())?, dim(d.conv_kernel() - 1)?], &taps)?;
        batch.plain(vec![dim(d.conv_dim())?], &last)?;
        batch.plain(vec![dim(d.value_heads())?], &neg)?;
        batch.plain(vec![dim(d.value_heads())?], w.dt_bias)?;
        batch.plain(vec![dim(d.value_dim())?], w.norm)
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

/// QSA weights. The four norms hold `1 + w`: query and key `[D]`, indexer query and key `[ID]`.
pub(super) struct RQsa {
    pub(super) q: ResidentPackedWeight,
    pub(super) k: ResidentPackedWeight,
    pub(super) v: ResidentPackedWeight,
    pub(super) o: ResidentPackedWeight,
    pub(super) idx: ResidentPackedWeight,
    pub(super) q_norm1p: ExecutionTensor,
    pub(super) k_norm1p: ExecutionTensor,
    pub(super) idx_q_norm1p: ExecutionTensor,
    pub(super) idx_k_norm1p: ExecutionTensor,
}

impl RQsa {
    fn queue<'w>(batch: &mut Batch<'w>, w: &QsaWeights<'w, StoredWeight>) -> Result<(), EngError> {
        for weight in [w.q, w.k, w.v, w.o, w.idx] {
            batch.packed(weight)?;
        }
        for norm in [w.q_norm, w.k_norm, w.idx_q_norm, w.idx_k_norm] {
            batch.plain(vec![dim(norm.len())?], &one_plus(norm))?;
        }
        Ok(())
    }
    fn take(got: &mut Imported) -> Result<Self, EngError> {
        Ok(Self {
            q: got.packed()?,
            k: got.packed()?,
            v: got.packed()?,
            o: got.packed()?,
            idx: got.packed()?,
            q_norm1p: got.plain()?,
            k_norm1p: got.plain()?,
            idx_q_norm1p: got.plain()?,
            idx_k_norm1p: got.plain()?,
        })
    }
    fn bytes(&self, total: &mut Bytes) -> Result<(), EngError> {
        for w in [&self.q, &self.k, &self.v, &self.o, &self.idx] {
            total.packed(w)?;
        }
        for t in [
            &self.q_norm1p,
            &self.k_norm1p,
            &self.idx_q_norm1p,
            &self.idx_k_norm1p,
        ] {
            total.tensor(t)?;
        }
        Ok(())
    }
}

/// MoE weights: `router_t` `[H, E]` F32; expert stacks stay streamed.
pub(super) struct RMoe {
    pub(super) router_t: ExecutionTensor,
    pub(super) shared_gate: ResidentPackedWeight,
    pub(super) shared_up: ResidentPackedWeight,
    pub(super) shared_down: ResidentPackedWeight,
    pub(super) shared_scalar: ResidentPackedWeight,
    pub(super) gate: SlicedWeight,
    pub(super) up: SlicedWeight,
    pub(super) down: SlicedWeight,
}

impl RMoe {
    fn queue<'w>(
        batch: &mut Batch<'w>,
        w: &MoeWeights<'w, StoredWeight>,
        shape: &ModelShape,
    ) -> Result<(), EngError> {
        let (experts, hidden) = (shape.moe.experts(), shape.hyper.hidden());
        batch.plain(
            vec![dim(hidden)?, dim(experts)?],
            &transpose(w.router, experts, hidden)?,
        )?;
        for weight in [w.shared_gate, w.shared_up, w.shared_down, w.shared_scalar] {
            batch.packed(weight)?;
        }
        Ok(())
    }
    fn take(got: &mut Imported, w: &MoeWeights<'_, StoredWeight>) -> Result<Self, EngError> {
        Ok(Self {
            router_t: got.plain()?,
            shared_gate: got.packed()?,
            shared_up: got.packed()?,
            shared_down: got.packed()?,
            shared_scalar: got.packed()?,
            gate: sliced(w.gate)?,
            up: sliced(w.up)?,
            down: sliced(w.down)?,
        })
    }
    fn bytes(&self, total: &mut Bytes) -> Result<(), EngError> {
        total.tensor(&self.router_t)?;
        for w in [
            &self.shared_gate,
            &self.shared_up,
            &self.shared_down,
            &self.shared_scalar,
        ] {
            total.packed(w)?;
        }
        Ok(())
    }
}

/// PLE weights: norms `[C, H]` hold `1 + w`; `conv_taps` `[W, PCK - 1]`, `conv_last` `[W]`; tables stay streamed.
pub(super) struct RPle {
    pub(super) multipliers: Vec<i64>,
    pub(super) vocab: Vec<i64>,
    pub(super) offsets: Vec<i64>,
    pub(super) shards: Vec<SlicedWeight>,
    pub(super) shard_rows: Vec<u64>,
    pub(super) key: ResidentPackedWeight,
    pub(super) value: ResidentPackedWeight,
    pub(super) norm_key1p: ExecutionTensor,
    pub(super) norm_query1p: ExecutionTensor,
    pub(super) norm_conv1p: ExecutionTensor,
    pub(super) conv_taps: ExecutionTensor,
    pub(super) conv_last: ExecutionTensor,
}

impl RPle {
    fn queue<'w>(
        batch: &mut Batch<'w>,
        w: &PleWeights<'w, StoredWeight>,
        shape: &ModelShape,
    ) -> Result<(), EngError> {
        let rows = vec![dim(shape.hyper.branches())?, dim(shape.hyper.hidden())?];
        let (width, kernel) = (shape.hyper.width(), shape.ple.dims().conv_kernel as usize);
        let (taps, last) = split_conv(w.conv, width, kernel)?;
        batch.packed(w.key)?;
        batch.packed(w.value)?;
        for norm in [w.norm_key, w.norm_query, w.norm_conv] {
            batch.plain(rows.clone(), &one_plus(norm))?;
        }
        batch.plain(vec![dim(width)?, dim(kernel - 1)?], &taps)?;
        batch.plain(vec![dim(width)?], &last)
    }
    fn take(got: &mut Imported, w: &PleWeights<'_, StoredWeight>) -> Result<Self, EngError> {
        Ok(Self {
            multipliers: w.multipliers.to_vec(),
            vocab: w.vocab.to_vec(),
            offsets: w.offsets.to_vec(),
            shards: w.shards.iter().map(sliced).collect::<Result<Vec<_>, _>>()?,
            shard_rows: w.shard_rows.to_vec(),
            key: got.packed()?,
            value: got.packed()?,
            norm_key1p: got.plain()?,
            norm_query1p: got.plain()?,
            norm_conv1p: got.plain()?,
            conv_taps: got.plain()?,
            conv_last: got.plain()?,
        })
    }
    fn bytes(&self, total: &mut Bytes) -> Result<(), EngError> {
        total.packed(&self.key)?;
        total.packed(&self.value)?;
        for t in [
            &self.norm_key1p,
            &self.norm_query1p,
            &self.norm_conv1p,
            &self.conv_taps,
            &self.conv_last,
        ] {
            total.tensor(t)?;
        }
        Ok(())
    }
}

/// Attention weights of one layer.
pub(super) enum RAttn {
    Delta(RDelta),
    Qsa(RQsa),
}

/// Weights of one decoder layer.
pub(super) struct RLayer {
    pub(super) attn_hyper: RHyper,
    pub(super) attn: RAttn,
    pub(super) mlp_hyper: RHyper,
    pub(super) moe: RMoe,
}

/// Resident Flash-Next weights. The caller drops `FlashWeights` after `import`, so host bytes are not held twice.
pub(super) struct ResidentWeights {
    pub(super) shape: ModelShape,
    pub(super) embed: SlicedWeight,
    pub(super) lm_head: ResidentPackedWeight,
    pub(super) mixer: RHyper,
    pub(super) layers: Vec<RLayer>,
    pub(super) ple: RPle,
    pub(super) consts: Consts,
}

impl ResidentWeights {
    /// Imports with `layers + 2` synchronizations: head, mixer, and PLE; one per layer; the constants.
    pub(super) fn import(
        kernels: &mut MlxKernels,
        weights: &FlashWeights,
    ) -> Result<Self, EngError> {
        let shape = *weights.shape();
        let view = weights.view();
        let mut batch = Batch::default();
        batch.packed(view.lm_head)?;
        RHyper::queue(&mut batch, &view.mixer, &shape)?;
        RPle::queue(&mut batch, &view.ple, &shape)?;
        let mut got = batch.import(kernels)?;
        let lm_head = got.packed()?;
        let mixer = RHyper::take(&mut got, view.mixer.inject.is_some())?;
        let ple = RPle::take(&mut got, &view.ple)?;
        got.finish()?;
        let mut layers = Vec::with_capacity(view.layers.len());
        for layer in &view.layers {
            let mut batch = Batch::default();
            RHyper::queue(&mut batch, &layer.attn_hyper, &shape)?;
            match &layer.attn {
                AttnWeights::Delta(w) => RDelta::queue(&mut batch, w, &shape)?,
                AttnWeights::Qsa(w) => RQsa::queue(&mut batch, w)?,
            }
            RHyper::queue(&mut batch, &layer.mlp_hyper, &shape)?;
            RMoe::queue(&mut batch, &layer.moe, &shape)?;
            let mut got = batch.import(kernels)?;
            let attn_hyper = RHyper::take(&mut got, layer.attn_hyper.inject.is_some())?;
            let attn = match &layer.attn {
                AttnWeights::Delta(_) => RAttn::Delta(RDelta::take(&mut got)?),
                AttnWeights::Qsa(_) => RAttn::Qsa(RQsa::take(&mut got)?),
            };
            let mlp_hyper = RHyper::take(&mut got, layer.mlp_hyper.inject.is_some())?;
            let moe = RMoe::take(&mut got, &layer.moe)?;
            got.finish()?;
            layers.push(RLayer {
                attn_hyper,
                attn,
                mlp_hyper,
                moe,
            });
        }
        let consts = Consts::import(kernels, &shape, view.ple.weight_scale)?;
        Ok(Self {
            shape,
            embed: sliced(view.embed)?,
            lm_head,
            mixer,
            layers,
            ple,
            consts,
        })
    }

    /// Resident bytes of all weights and constants.
    pub(super) fn byte_len(&self) -> Result<u64, EngError> {
        let mut total = Bytes::default();
        total.packed(&self.lm_head)?;
        self.mixer.bytes(&mut total)?;
        self.ple.bytes(&mut total)?;
        for layer in &self.layers {
            layer.attn_hyper.bytes(&mut total)?;
            match &layer.attn {
                RAttn::Delta(d) => d.bytes(&mut total)?,
                RAttn::Qsa(q) => q.bytes(&mut total)?,
            }
            layer.mlp_hyper.bytes(&mut total)?;
            layer.moe.bytes(&mut total)?;
        }
        self.consts.bytes(&mut total)?;
        Ok(total.total())
    }
}

/// Imports the synthetic checkpoint of `fixtures` (test helper).
#[cfg(test)]
pub(super) fn synthetic(kernels: &mut MlxKernels) -> ResidentWeights {
    let ckpt = super::fixtures::checkpoint();
    let flash = FlashWeights::load(&ckpt, &super::fixtures::config()).unwrap();
    ResidentWeights::import(kernels, &flash).unwrap()
}

#[cfg(test)]
mod tests {
    use super::super::ops::{assert_close, values};
    use super::super::state::ResidentState;
    use super::*;
    use crate::weights::{ShardIo, StreamedCheckpoint};
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};

    #[test]
    fn import_keeps_block_shapes_and_counts_bytes() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = super::synthetic(&mut kernels);
        assert_eq!(weights.layers.len(), 2, "layer count");
        let RAttn::Delta(delta) = &weights.layers[0].attn else {
            panic!("layer 0 is not DeltaNet");
        };
        let RAttn::Qsa(qsa) = &weights.layers[1].attn else {
            panic!("layer 1 is not QSA");
        };
        assert_eq!(weights.lm_head.logical_shape(), &[8, 64][..], "lm head");
        assert!(weights.mixer.inject.is_none(), "mixer has no inject");
        assert!(
            weights.layers[0].attn_hyper.inject.is_some(),
            "layer hyper has inject"
        );
        assert_eq!(weights.mixer.norm1p.shape(), &[2, 64][..], "mixer norm1p");
        assert_eq!(delta.qkv.logical_shape(), &[128, 64][..], "delta qkv");
        assert_eq!(delta.conv_taps.shape(), &[128, 3][..], "delta conv taps");
        assert_eq!(delta.conv_last.shape(), &[128][..], "delta conv last");
        assert_eq!(delta.norm.shape(), &[32][..], "delta norm");
        assert_eq!(qsa.q.logical_shape(), &[256, 64][..], "qsa q");
        assert_eq!(qsa.idx_q_norm1p.shape(), &[32][..], "qsa indexer q norm");
        assert_eq!(
            weights.layers[0].moe.router_t.shape(),
            &[64, 4][..],
            "router transpose"
        );
        assert_eq!(
            weights.layers[0].moe.shared_scalar.logical_shape(),
            &[1, 64][..],
            "shared scalar"
        );
        assert_eq!(weights.ple.norm_key1p.shape(), &[2, 64][..], "ple norm key");
        assert_eq!(
            weights.ple.conv_taps.shape(),
            &[128, 3][..],
            "ple conv taps"
        );
        assert_eq!(weights.ple.shards.len(), 1, "ple shard count");
        assert_eq!(weights.ple.key.logical_shape(), &[128, 64][..], "ple key");
        assert_eq!(weights.byte_len().unwrap(), 152_172, "resident bytes");
    }

    #[test]
    fn import_precomputes_norms_router_and_conv() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = super::synthetic(&mut kernels);
        assert_close(
            &values(&mut kernels, &weights.mixer.norm1p).unwrap(),
            &vec![1.5; 128],
            "mixer norm1p",
        );
        let RAttn::Delta(delta) = &weights.layers[0].attn else {
            panic!("layer 0 is not DeltaNet");
        };
        assert_close(
            &values(&mut kernels, &delta.neg_exp_a_log).unwrap(),
            &[-1.0, -std::f32::consts::E],
            "delta neg exp a log",
        );
        assert_close(
            &values(&mut kernels, &delta.dt_bias).unwrap(),
            &[0.5, -0.5],
            "delta dt bias",
        );
        assert_close(
            &values(&mut kernels, &delta.norm).unwrap(),
            &[1.0; 32],
            "delta norm",
        );
        let taps: Vec<f32> = (0..128)
            .flat_map(|c| (0..3).map(move |k| (4 * c + k) as f32))
            .collect();
        assert_close(
            &values(&mut kernels, &delta.conv_taps).unwrap(),
            &taps,
            "delta conv taps",
        );
        let last: Vec<f32> = (0..128).map(|c| (4 * c + 3) as f32).collect();
        assert_close(
            &values(&mut kernels, &delta.conv_last).unwrap(),
            &last,
            "delta conv last",
        );
        let router: Vec<f32> = (0..64)
            .flat_map(|h| (0..4).map(move |e| (64 * e + h) as f32))
            .collect();
        assert_close(
            &values(&mut kernels, &weights.layers[0].moe.router_t).unwrap(),
            &router,
            "router transpose",
        );
        let RAttn::Qsa(qsa) = &weights.layers[1].attn else {
            panic!("layer 1 is not QSA");
        };
        assert_close(
            &values(&mut kernels, &qsa.q_norm1p).unwrap(),
            &vec![1.25; 64],
            "qsa q norm1p",
        );
        assert_close(
            &values(&mut kernels, &qsa.k_norm1p).unwrap(),
            &vec![0.75; 64],
            "qsa k norm1p",
        );
        assert_close(
            &values(&mut kernels, &qsa.idx_k_norm1p).unwrap(),
            &[1.0; 32],
            "qsa indexer k norm1p",
        );
        assert_close(
            &values(&mut kernels, &weights.ple.norm_conv1p).unwrap(),
            &vec![1.5; 128],
            "ple norm conv1p",
        );
        assert_close(
            &values(&mut kernels, &weights.ple.conv_taps).unwrap(),
            &taps,
            "ple conv taps",
        );
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn import_uses_one_synchronization_per_layer_plus_two() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = super::super::fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &super::super::fixtures::config()).unwrap();
        let before = kernels.qualification_sync_count();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before + 4,
            "globals, 2 layers, constants"
        );
        let state = ResidentState::zero(&mut kernels, &weights).unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before + 5,
            "zero state adds one synchronization"
        );
        assert_eq!(state.pos(), 0, "state position");
    }

    /// Returns the `weight_map` object of a safetensors index.
    fn weight_map(index: &[u8]) -> serde_json::Map<String, serde_json::Value> {
        let value: serde_json::Value = serde_json::from_slice(index).unwrap();
        value["weight_map"].as_object().unwrap().clone()
    }

    fn open_checkpoint(dir: &Path) -> StreamedCheckpoint {
        let index = std::fs::read(dir.join("model.safetensors.index.json")).unwrap();
        let names: BTreeSet<String> = weight_map(&index)
            .values()
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect();
        let shards: Vec<(String, Box<dyn ShardIo + Send>, u64)> = names
            .into_iter()
            .map(|name| {
                let file = std::fs::File::open(dir.join(&name)).unwrap();
                let len = file.metadata().unwrap().len();
                (name, Box::new(file) as Box<dyn ShardIo + Send>, len)
            })
            .collect();
        StreamedCheckpoint::open(&index, shards, 64 << 20).unwrap()
    }

    fn syncs(k: &MlxKernels) -> Option<u64> {
        #[cfg(feature = "native-fault-qualification")]
        {
            Some(k.qualification_sync_count())
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        {
            let _ = k;
            None
        }
    }

    #[test]
    #[ignore = "needs the Flash-Next artifact on the Mac"]
    fn real_checkpoint_import_reports_resident_bytes() {
        let _serial = crate::native_test_lock();
        let dir = PathBuf::from(
            std::env::var("FLASH_NEXT_ARTIFACT").expect("FLASH_NEXT_ARTIFACT is not set"),
        );
        let report = PathBuf::from(
            std::env::var("FLASH_NEXT_RESIDENT_REPORT")
                .expect("FLASH_NEXT_RESIDENT_REPORT is not set"),
        );
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = open_checkpoint(&dir);
        let config = std::fs::read(dir.join("config.json")).unwrap();
        let t = std::time::Instant::now();
        let flash = FlashWeights::load(&ckpt, &config).unwrap();
        let load_seconds = t.elapsed().as_secs_f64();
        kernels.reset_peak_memory().unwrap();
        let before = syncs(&kernels);
        let t = std::time::Instant::now();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let import_seconds = t.elapsed().as_secs_f64();
        let after_import = syncs(&kernels);
        let stats = kernels.memory_stats().unwrap();
        let (active_after_import, peak_after_import, cache_after_import) =
            (stats.active, stats.peak, stats.cache);
        drop(flash);
        let active_after_drop = kernels.memory_stats().unwrap().active;
        let state = ResidentState::zero(&mut kernels, &weights).unwrap();
        let after_zero = syncs(&kernels);
        let weights_bytes = weights.byte_len().unwrap();
        let router_bytes: u64 = weights
            .layers
            .iter()
            .map(|layer| layer.moe.router_t.byte_len())
            .sum();
        let state_bytes = state.byte_len().unwrap();
        let import_syncs = after_import
            .zip(before)
            .map(|(after, before)| after - before);
        let zero_syncs = after_zero
            .zip(after_import)
            .map(|(after, before)| after - before);
        let other_bytes = weights_bytes - router_bytes;
        let lines = format!(
            "load_seconds={load_seconds}\nimport_seconds={import_seconds}\nweights_bytes={weights_bytes}\n\
             router_bytes={router_bytes}\nother_bytes={other_bytes}\nstate_bytes={state_bytes}\n\
             active_after_import={active_after_import}\npeak_after_import={peak_after_import}\n\
             cache_after_import={cache_after_import}\nactive_after_drop={active_after_drop}\n\
             import_syncs={import_syncs:?}\nzero_syncs={zero_syncs:?}\n"
        );
        std::fs::write(&report, lines).unwrap();
        assert_eq!(weights.layers.len(), 48, "layer count");
        assert_eq!(router_bytes, 251_658_240, "router bytes");
        assert_eq!(state_bytes, 118_038_528, "state bytes");
        if let (Some(before), Some(after)) = (before, after_import) {
            assert_eq!(after - before, 50, "import synchronizations");
        }
        if let (Some(after_import), Some(after_zero)) = (after_import, after_zero) {
            assert_eq!(after_zero - after_import, 1, "zero state synchronizations");
        }
    }
}
