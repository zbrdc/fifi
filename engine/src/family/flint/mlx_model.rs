//! Bounded Flint weight import. Only resident packed matrices survive each import.

use super::{
    mlx_expert_cache::{self, LayerExpertCache},
    mlx_math::{self, Projection, SwiGluWeights},
    weights::FlintCheckpoint,
};
use crate::{
    EngError,
    family::prefill::mlx::{overlap_expert_reads, prefill_expert_read_runs},
    kernels::{HostPages, MlxKernels, PackedPages, PageRange, ResidentPackedWeight},
    tensor::{Dtype, ExecutionTensor, Tensor},
};
use std::{cell::RefCell, path::Path, rc::Rc, sync::OnceLock};

pub(super) fn expert_pipeline_flagged() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("FIFI_FLINT_MLX_EXPERT_PIPELINE").map_or_else(
            || overlap_expert_reads() || prefill_expert_read_runs(),
            |value| value == "1",
        )
    })
}

#[derive(Clone, Copy)]
pub(super) enum ExpertReadRuns {
    Off,
    Prefill,
    All,
}

impl ExpertReadRuns {
    fn from_env(value: Option<&str>) -> Self {
        match value {
            Some("prefill") => Self::Prefill,
            Some("1") => Self::All,
            _ => Self::Off,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Prefill => "prefill",
            Self::All => "on",
        }
    }

    pub(super) fn coalesce(self, rows: u32) -> bool {
        matches!(self, Self::All) || (matches!(self, Self::Prefill) && rows > 1)
    }
}

pub(super) fn expert_read_runs() -> ExpertReadRuns {
    static MODE: OnceLock<ExpertReadRuns> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("FIFI_FLINT_MLX_EXPERT_READ_RUNS") {
        Ok(value) => ExpertReadRuns::from_env(Some(&value)),
        Err(std::env::VarError::NotPresent) if prefill_expert_read_runs() => {
            ExpertReadRuns::Prefill
        }
        Err(_) => ExpertReadRuns::Off,
    })
}

#[cfg(test)]
#[test]
fn expert_read_runs_preserve_legacy_and_allow_multi_row_only() {
    for (value, one, many) in [
        (None, false, false),
        (Some("0"), false, false),
        (Some("unexpected"), false, false),
        (Some("prefill"), false, true),
        (Some("1"), true, true),
    ] {
        let mode = ExpertReadRuns::from_env(value);
        assert_eq!(mode.coalesce(1), one, "{value:?}");
        assert_eq!(mode.coalesce(2), many, "{value:?}");
    }
}

/// Mac MLX freed-buffer cache limit. Default 64 MiB: bounded reuse keeps
/// per-step freed buffers on the device instead of paying a driver
/// round-trip for every free/malloc pair. Rollback:
/// FIFI_FLINT_MLX_CACHE_LIMIT=0 (read once per process at first use;
/// changes require a restart); an unrecognized SET value fails safe to the
/// previous 0 limit, not 64 MiB.
pub(super) fn cache_limit_bytes() -> u64 {
    use std::sync::OnceLock;
    static LIMIT: OnceLock<u64> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        let limit = super::factory::cache_limit_with(super::factory::os_setting(
            std::env::var_os("FIFI_FLINT_MLX_CACHE_LIMIT").as_deref(),
        ));
        // The one load line an operator needs to confirm the effective limit.
        eprintln!(
            "flint: MLX freed-buffer cache {} MiB",
            limit / (1024 * 1024)
        );
        limit
    })
}

type Result<T> = std::result::Result<T, EngError>;

pub(super) struct SwiGlu {
    pub(super) gate: ResidentPackedWeight,
    pub(super) up: ResidentPackedWeight,
    pub(super) down: ResidentPackedWeight,
}

impl SwiGlu {
    fn load(k: &mut MlxKernels, source: &FlintCheckpoint, stem: &str) -> Result<Self> {
        Ok(Self {
            gate: packed(k, source, &format!("{stem}.gate_proj"))?,
            up: packed(k, source, &format!("{stem}.up_proj"))?,
            down: packed(k, source, &format!("{stem}.down_proj"))?,
        })
    }

    pub(super) fn load_experts(
        k: &mut MlxKernels,
        source: &FlintCheckpoint,
        stem: &str,
        experts: &[u32],
        remapped: &Tensor,
    ) -> Result<(Self, ExecutionTensor)> {
        let gate = source.packed_experts(&format!("{stem}.gate_proj"), experts)?;
        let up = source.packed_experts(&format!("{stem}.up_proj"), experts)?;
        let down = source.packed_experts(&format!("{stem}.down_proj"), experts)?;
        let (weights, ids) = k.import_packed_many_with(&[&gate, &up, &down], &[remapped])?;
        Self::take_experts(weights, ids)
    }

    pub(super) fn load_expert_pages(
        k: &mut MlxKernels,
        source: &FlintCheckpoint,
        stem: &str,
        experts: &[u32],
        remapped: &Tensor,
        coalesce_reads: bool,
    ) -> Result<(Self, ExecutionTensor, HostPages)> {
        let (pages, packed, local) =
            source.expert_pages(stem, experts, remapped, coalesce_reads)?;
        let (weights, ids) = Self::import_expert_pages(k, &pages, &packed, local)?;
        Ok((weights, ids, pages))
    }

    pub(super) fn import_expert_pages(
        k: &mut MlxKernels,
        pages: &HostPages,
        packed: &[PackedPages],
        local: PageRange,
    ) -> Result<(Self, ExecutionTensor)> {
        #[cfg(feature = "native-fault-qualification")]
        let started = super::mlx_expert_timing::enabled().then(std::time::Instant::now);
        let (weights, ids) = k.import_packed_pages(pages, packed, &[local])?;
        #[cfg(feature = "native-fault-qualification")]
        if let Some(started) = started {
            crate::family::stored::record_page_import(started.elapsed());
        }
        Self::take_experts(weights, ids)
    }

    fn take_experts(
        weights: Vec<ResidentPackedWeight>,
        ids: Vec<ExecutionTensor>,
    ) -> Result<(Self, ExecutionTensor)> {
        let [gate, up, down] = weights
            .try_into()
            .map_err(|_| EngError::Checkpoint("short Flint expert import".into()))?;
        let [ids] = ids
            .try_into()
            .map_err(|_| EngError::Checkpoint("short Flint expert ID import".into()))?;
        Ok((Self { gate, up, down }, ids))
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

pub(super) enum Routed {
    Resident(SwiGlu),
    Disk {
        source: Rc<FlintCheckpoint>,
        stem: String,
        cache: Option<Box<RefCell<LayerExpertCache>>>,
    },
}

pub(super) enum Ffn {
    Dense(SwiGlu),
    Moe {
        router: ExecutionTensor,
        routed: Routed,
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
    pub(super) fn load(
        k: &mut MlxKernels,
        source: &Rc<FlintCheckpoint>,
        index: usize,
        disk: bool,
        cache: Option<LayerExpertCache>,
    ) -> Result<Self> {
        let stem = format!("model.layers.{index}");
        let ffn = if index < 2 {
            Ffn::Dense(SwiGlu::load(k, source, &format!("{stem}.mlp"))?)
        } else {
            let router = plain(k, source, &format!("{stem}.mlp.gate.weight"))?;
            let router = k.transpose(&router, &[1, 0])?;
            k.evaluate(&[&router])?;
            Ffn::Moe {
                router,
                routed: if disk {
                    Routed::Disk {
                        source: Rc::clone(source),
                        stem: format!("{stem}.mlp.switch_mlp"),
                        cache: cache.map(|cache| Box::new(RefCell::new(cache))),
                    }
                } else {
                    Routed::Resident(SwiGlu::load(k, source, &format!("{stem}.mlp.switch_mlp"))?)
                },
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
    pub(super) source: Rc<FlintCheckpoint>,
    pub(super) streamed_expert_bytes: u64,
    pub(super) cached_expert_bytes: u64,
    pub(super) kernels: MlxKernels,
}

impl Model {
    #[cfg(test)]
    pub(super) fn open(root: &Path) -> Result<Self> {
        Self::open_for_budget(root, None)
    }

    pub(super) fn open_for_budget(root: &Path, available: Option<u64>) -> Result<Self> {
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
        // Default-off neutrality: the budget-off arm keeps main's text
        // verbatim; the reworded line prints only when device sizing is on.
        let ceiling = super::factory::prefill_chunk_rows(super::factory::os_setting(
            std::env::var_os("FIFI_FLINT_MLX_BATCH_PREFILL").as_deref(),
        ));
        if crate::family::mlx_budget::enabled() {
            eprintln!("flint: MLX configured prefill ceiling {ceiling} rows");
        } else {
            eprintln!("flint: MLX prefill piece size {ceiling} rows");
        }
        let source = Rc::new(FlintCheckpoint::open(root)?);
        let streamed_expert_bytes = match available {
            Some(bytes) => plan_weight_placement(&source, bytes)?,
            None => 0,
        };
        let disk = streamed_expert_bytes != 0;
        let (caches, cached_expert_bytes) = if let Some(budget) = available.filter(|_| disk) {
            let cache_budget = budget
                .saturating_sub(source.mlx_weight_bytes(true))
                .saturating_sub(expert_staging_bytes(streamed_expert_bytes, 1));
            mlx_expert_cache::plan(&source, cache_budget)?
        } else {
            ((0..super::config::LAYERS).map(|_| None).collect(), 0)
        };
        #[cfg(feature = "native-fault-qualification")]
        if let Some(description) = source.expert_record_description() {
            if !disk || !expert_pipeline_flagged() || cached_expert_bytes != 0 {
                return Err(EngError::BadRequest(
                    "Flint expert records need disk-backed experts, the shared-page pipeline and expert caching off".into(),
                ));
            }
            eprintln!("flint: MLX expert records {description}");
        }
        if disk {
            let pipeline = expert_pipeline_flagged();
            eprintln!(
                "flint: MLX expert pipeline {}, contiguous reads {}",
                if pipeline { "on" } else { "off" },
                if pipeline {
                    expert_read_runs().label()
                } else {
                    "off"
                },
            );
        }
        if let Some(bytes) = available {
            eprintln!(
                "flint: MLX weight placement={} weight_budget_bytes={bytes} resident_estimate_bytes={} streamed_bytes_per_expert={streamed_expert_bytes}",
                if disk { "disk-experts" } else { "resident" },
                source.mlx_weight_bytes(disk),
            );
        }
        let mut kernels = MlxKernels::try_new()?;
        kernels.allow_bf16(true);
        // Bounded freed-buffer reuse: without a cache every per-step
        // free/malloc pair pays a driver round-trip. 64 MiB covers the
        // measured decode churn envelope.
        kernels.set_cache_limit(cache_limit_bytes())?;
        let mut layers = Vec::with_capacity(super::config::LAYERS);
        for (index, cache) in caches.into_iter().enumerate() {
            layers.push(Layer::load(&mut kernels, &source, index, disk, cache)?);
        }
        let norm = plain(&mut kernels, &source, "model.norm.weight")?;
        let head = packed(&mut kernels, &source, "lm_head")?;
        Ok(Self {
            layers,
            norm,
            head,
            source,
            streamed_expert_bytes,
            cached_expert_bytes,
            kernels,
        })
    }
}

pub(super) fn expert_staging_bytes(bytes_per_expert: u64, rows: u32) -> u64 {
    2 * bytes_per_expert * u64::from(rows.saturating_mul(6).min(256))
}

fn plan_weight_placement(source: &FlintCheckpoint, available: u64) -> Result<u64> {
    if source.mlx_weight_bytes(false) <= available {
        return Ok(0);
    }
    let per_expert = source.bytes_per_routed_expert();
    let minimum = source.mlx_weight_bytes(true) + expert_staging_bytes(per_expert, 1);
    if minimum > available {
        return Err(EngError::MemoryPlan(format!(
            "Flint disk-backed experts require {minimum} bytes for resident weights and one token of staging, with {available} bytes left after serving reserves"
        )));
    }
    Ok(per_expert)
}

fn packed(
    k: &mut MlxKernels,
    source: &FlintCheckpoint,
    stem: &str,
) -> Result<ResidentPackedWeight> {
    let host = source.packed(stem)?;
    k.import_packed(&host)
}

fn plain(k: &mut MlxKernels, source: &FlintCheckpoint, name: &str) -> Result<ExecutionTensor> {
    let host = source.tensor(name)?;
    // Preserve publisher BF16 norm/router parameters; the sink alone is
    // widened at its layer load site, then rounded to query dtype at use.
    k.import_tensor(&host)
}
