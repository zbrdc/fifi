//! Bounded Flint weight import. Only resident packed matrices survive each import.

use super::{
    mlx_math::{self, Projection, SwiGluWeights},
    weights::FlintCheckpoint,
};
use crate::{
    EngError,
    kernels::{MlxKernels, ResidentPackedWeight},
    tensor::{Dtype, ExecutionTensor},
};
use std::path::Path;

type Result<T> = std::result::Result<T, EngError>;

pub(super) struct SwiGlu {
    pub(super) gate: ResidentPackedWeight,
    pub(super) up: ResidentPackedWeight,
    pub(super) down: ResidentPackedWeight,
}

impl SwiGlu {
    fn load(k: &mut MlxKernels, source: &mut FlintCheckpoint, stem: &str) -> Result<Self> {
        Ok(Self {
            gate: packed(k, source, &format!("{stem}.gate_proj"))?,
            up: packed(k, source, &format!("{stem}.up_proj"))?,
            down: packed(k, source, &format!("{stem}.down_proj"))?,
        })
    }

    pub(super) fn apply(&self, k: &mut MlxKernels, x: &ExecutionTensor) -> Result<ExecutionTensor> {
        mlx_math::expert(
            k,
            &SwiGluWeights {
                gate: Projection::Packed(&self.gate),
                up: Projection::Packed(&self.up),
                down: Projection::Packed(&self.down),
            },
            x,
        )
    }
}

pub(super) enum Ffn {
    Dense(SwiGlu),
    Moe {
        router: ExecutionTensor,
        routed: SwiGlu,
        shared: Box<SwiGlu>,
    },
}

pub(super) struct Layer {
    pub(super) input_norm: ExecutionTensor,
    pub(super) post_norm: ExecutionTensor,
    pub(super) q_norm: ExecutionTensor,
    pub(super) k_norm: ExecutionTensor,
    pub(super) sink: ExecutionTensor,
    pub(super) q: ResidentPackedWeight,
    pub(super) k: ResidentPackedWeight,
    pub(super) v: ResidentPackedWeight,
    pub(super) o: ResidentPackedWeight,
    pub(super) ffn: Ffn,
}

impl Layer {
    fn load(k: &mut MlxKernels, source: &mut FlintCheckpoint, index: usize) -> Result<Self> {
        let stem = format!("model.layers.{index}");
        let ffn = if index < 2 {
            Ffn::Dense(SwiGlu::load(k, source, &format!("{stem}.mlp"))?)
        } else {
            let router = plain(k, source, &format!("{stem}.mlp.gate.weight"))?;
            let router = k.transpose(&router, &[1, 0])?;
            k.evaluate(&[&router])?;
            Ffn::Moe {
                router,
                routed: SwiGlu::load(k, source, &format!("{stem}.mlp.switch_mlp"))?,
                shared: Box::new(SwiGlu::load(
                    k,
                    source,
                    &format!("{stem}.mlp.shared_expert"),
                )?),
            }
        };
        Ok(Self {
            input_norm: plain(k, source, &format!("{stem}.input_layernorm.weight"))?,
            post_norm: plain(
                k,
                source,
                &format!("{stem}.post_attention_layernorm.weight"),
            )?,
            q_norm: plain(k, source, &format!("{stem}.self_attn.q_norm.weight"))?,
            k_norm: plain(k, source, &format!("{stem}.self_attn.k_norm.weight"))?,
            sink: {
                let sink = plain(k, source, &format!("{stem}.self_attn.softmax_offset"))?;
                let sink = k.cast(&sink, Dtype::F32)?;
                // The vendor decode route wants sinks in the output dtype;
                // on a flagged load they are stored AS BF16 (the exactness
                // check below refuses a checkpoint whose values would
                // round, so the stored bits ARE the published logits).
                let sink = if crate::kernels::mlx::decode_attn_flagged() {
                    let host = k.readback(&sink)?;
                    let bytes = host.host_bytes().ok_or_else(|| {
                        EngError::BadRequest(format!("{stem}: sink readback has no host bytes"))
                    })?;
                    let values: Vec<f32> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|w| f32::from_le_bytes(*w))
                        .collect();
                    crate::kernels::mlx::assert_sinks_bf16_exact(&values, &stem)?;
                    k.cast(&sink, Dtype::Bf16)?
                } else {
                    sink
                };
                // The shared API takes raw query-head sinks, never a GQA fold.
                let sink = k.reshape(&sink, &[1, 32])?;
                k.evaluate(&[&sink])?;
                sink
            },
            q: packed(k, source, &format!("{stem}.self_attn.q_proj"))?,
            k: packed(k, source, &format!("{stem}.self_attn.k_proj"))?,
            v: packed(k, source, &format!("{stem}.self_attn.v_proj"))?,
            o: packed(k, source, &format!("{stem}.self_attn.o_proj"))?,
            ffn,
        })
    }
}

/// All fields stay on one worker thread. Resident fields drop before the adapter.
pub(super) struct Model {
    pub(super) layers: Vec<Layer>,
    pub(super) norm: ExecutionTensor,
    pub(super) head: ResidentPackedWeight,
    pub(super) source: FlintCheckpoint,
    pub(super) kernels: MlxKernels,
}

impl Model {
    pub(super) fn open(root: &Path) -> Result<Self> {
        // Freeze and log the family's default-on switches at engine start
        // (one load line each; the OnceLocks keep every later read identical).
        let _ = super::mlx_forward::async_chain_flagged();
        let _ = super::mlx_forward::coalesce_residual_flagged();
        // Under the 8192 rollback the KV delta is zero, so the "up to"
        // clause would read "+0.000 GiB" - print the legacy line instead.
        if super::factory::context_limit() > 8192 {
            eprintln!(
                "flint: MLX context limit {} tokens (full-attention KV up to +{:.3} GiB over 8192 at full depth)",
                super::factory::context_limit(),
                // Extra full-attention KV vs the 8192 baseline (14 planes);
                // planes grow on demand, so this is the 32K-depth ceiling,
                // not an open-time allocation.
                // 14 full-attention planes x 2048 B/row (K+V) per extra row.
                (super::factory::context_limit().saturating_sub(8192) as f64 * 28672.0)
                    / (1024.0 * 1024.0 * 1024.0),
            );
        } else {
            eprintln!(
                "flint: MLX context limit {} tokens (legacy 8192 baseline)",
                super::factory::context_limit()
            );
        }
        eprintln!(
            "flint: MLX prefill piece size {} rows",
            super::factory::prefill_chunk_rows(super::factory::os_setting(
                std::env::var_os("FIFI_FLINT_MLX_BATCH_PREFILL").as_deref(),
            ))
        );
        let mut source = FlintCheckpoint::open(root)?;
        let mut kernels = MlxKernels::try_new()?;
        kernels.allow_bf16(true);
        // Initial synchronous path does not need MLX's unbounded freed-buffer cache.
        kernels.set_cache_limit(0)?;
        let mut layers = Vec::with_capacity(49);
        for index in 0..49 {
            layers.push(Layer::load(&mut kernels, &mut source, index)?);
        }
        let norm = plain(&mut kernels, &mut source, "model.norm.weight")?;
        let head = packed(&mut kernels, &mut source, "lm_head")?;
        Ok(Self {
            layers,
            norm,
            head,
            source,
            kernels,
        })
    }
}

fn packed(
    k: &mut MlxKernels,
    source: &mut FlintCheckpoint,
    stem: &str,
) -> Result<ResidentPackedWeight> {
    let host = source.packed(stem)?;
    k.import_packed(&host)
}

fn plain(k: &mut MlxKernels, source: &mut FlintCheckpoint, name: &str) -> Result<ExecutionTensor> {
    let host = source.tensor(name)?;
    // Preserve publisher BF16 norm/router parameters; the sink alone is
    // widened at its layer load site, then rounded to query dtype at use.
    k.import_tensor(&host)
}
