//! File-backed expert RAM tier. Warm admitted layers once, then use the shared
//! router predictor to request the next visit's pages ahead of CUDA demand.
use std::{cell::RefCell, collections::BTreeMap, ops::Range};

use super::{GgufCuda, GgufTextWeights, MappedFile, TextSpec, bad, expert_layer_order};
use crate::{
    EngError,
    family::{
        host_admit::HostAdmission,
        inkling2::cpu_forward::{route, sigmoid},
        qwen36::tiered::guess::{LayerRouter, RecallMeter, RouterGuess},
    },
};

pub(super) fn requested_limit() -> Result<Option<u64>, EngError> {
    const NAME: &str = "FIFI_INKLING_CUDA_RAM_EXPERT_MIB";
    let Some(raw) = std::env::var_os(NAME) else {
        return Ok(None);
    };
    if raw == "auto" {
        return Ok(Some(u64::MAX));
    }
    raw.to_str()
        .and_then(|s| s.parse::<u64>().ok())
        .and_then(|n| n.checked_mul(1 << 20))
        .map(Some)
        .ok_or_else(|| bad(format!("{NAME} must be auto or a nonnegative MiB count")))
}

pub(super) struct Stack {
    pub(super) layer: usize,
    parts: [Range<usize>; 2],
    experts: usize,
}

impl Stack {
    fn warm(&self, mapping: &MappedFile) {
        for part in &self.parts {
            let bytes = &mapping.as_ref()[part.clone()];
            // Fault each page without a second anonymous copy. Include the last
            // byte because an unaligned tensor can end on an extra page.
            for value in bytes.iter().step_by(4096).chain(bytes.last()) {
                std::hint::black_box(*value);
            }
        }
    }

    fn bytes(&self) -> u64 {
        self.parts.iter().map(|part| part.len() as u64).sum()
    }

    pub(super) fn ranges(&self, expert: usize) -> [Range<usize>; 2] {
        self.parts.each_ref().map(|part| {
            let stride = part.len() / self.experts;
            let start = part.start + expert * stride;
            start..start + stride
        })
    }
}

struct Prediction {
    stack: Stack,
    router: RouterGuess,
    bias: Vec<f32>,
    count: usize,
}

impl Prediction {
    fn load(
        weights: &GgufTextWeights,
        stack: Stack,
        hidden: usize,
        eps: f32,
        count: usize,
    ) -> Result<Self, EngError> {
        let experts = stack.experts;
        let prefix = format!("model.llm.layers.{}", stack.layer);
        let router = RouterGuess::new(
            vec![LayerRouter {
                router: weights
                    .vec(&format!("{prefix}.mlp.gate.weight"), hidden * experts)?
                    .into(),
                norm_weight: weights
                    .vec(&format!("{prefix}.mlp_norm.weight"), hidden)?
                    .into(),
            }],
            hidden,
            experts,
            eps,
        )
        .map_err(|e| bad(format!("GGUF RAM router: {e:?}")))?;
        let bias = weights.vec(&format!("{prefix}.mlp.gate.bias"), experts)?;
        Ok(Self {
            stack,
            router,
            bias,
            count,
        })
    }
}

pub(super) fn cold_stacks(
    weights: &GgufTextWeights,
    cuda: &GgufCuda,
    spec: &TextSpec,
) -> Result<Vec<Stack>, EngError> {
    let experts = spec.n_routed_experts as usize;
    let mut stacks = Vec::new();
    for layer in expert_layer_order(spec).filter(|&l| cuda.streams_layer(l as u32)) {
        let part = |key| -> Result<_, EngError> {
            Ok(weights
                .tensor(&format!(
                    "model.llm.layers.{layer}.mlp.experts.{key}_weight"
                ))?
                .0)
        };
        let parts = [part("w13")?, part("w2")?];
        if !parts.iter().all(|t| cuda.expert_dtype(t.kind).is_some()) {
            continue;
        }
        stacks.push(Stack {
            layer,
            parts: parts.map(|t| t.bytes.clone()),
            experts,
        });
    }
    Ok(stacks)
}

#[derive(Default)]
struct Stats {
    pending: Option<(usize, Vec<u16>)>,
    meter: RecallMeter,
    advice_bytes: u64,
    errors: u64,
}

pub(in crate::family::inkling2) struct RamExperts {
    predictive: BTreeMap<usize, Prediction>,
    stats: RefCell<Stats>,
}

impl RamExperts {
    pub(super) fn plan(
        weights: &GgufTextWeights,
        cuda: &GgufCuda,
        spec: &TextSpec,
        eps: f32,
        limit: u64,
        host_reserve: u64,
    ) -> Result<Option<Self>, EngError> {
        if limit == 0 {
            return Ok(None);
        }
        let admission = match HostAdmission::probe("Inkling expert RAM", "Inkling") {
            Ok(snapshot) => snapshot,
            Err(e) => {
                eprintln!("Inkling expert RAM prefetch unavailable: {e:?}");
                return Ok(None);
            }
        };
        let cap = limit.min(admission.cache_capacity().saturating_sub(host_reserve));
        let experts = spec.n_routed_experts as usize;
        let hidden = spec.hidden_size as usize;
        let k = spec.num_experts_per_tok as usize;
        let stacks = cold_stacks(weights, cuda, spec)?;
        let total: u64 = stacks.iter().map(Stack::bytes).sum();
        // Leave space for one speculative wave, including the decoded routers.
        let metadata = 4 * (hidden * experts + hidden + experts) as u64 * stacks.len() as u64;
        let wave = stacks
            .iter()
            .map(|s| s.bytes() / experts as u64 * k as u64)
            .max()
            .unwrap_or(0);
        let mut warm_budget = if total <= cap {
            cap
        } else {
            cap.saturating_sub(metadata.saturating_add(wave))
        };
        let window = cap.saturating_sub(warm_budget).saturating_sub(metadata);
        let mut warmed = 0;
        let mut predictive = BTreeMap::new();
        for stack in stacks {
            if stack.bytes() <= warm_budget {
                stack.warm(&weights.mapping);
                warm_budget -= stack.bytes();
                warmed += stack.bytes();
            } else {
                let count = k.min((window / (stack.bytes() / experts as u64)) as usize);
                if count == 0 {
                    continue;
                }
                predictive.insert(
                    stack.layer,
                    Prediction::load(weights, stack, hidden, eps, count)?,
                );
            }
        }
        eprintln!(
            "Inkling GGUF expert RAM {}",
            serde_json::json!({
                "admitted_bytes": cap, "non_gpu_expert_bytes": total, "warmed_bytes": warmed,
                "predictive_layers": predictive.len(), "max_advice_bytes_per_visit": window
            })
        );
        Ok((!predictive.is_empty()).then(|| Self {
            predictive,
            stats: RefCell::default(),
        }))
    }

    pub(in crate::family::inkling2) fn predict(
        &self,
        layer: usize,
        hidden: impl FnOnce() -> Vec<f32>,
        mapping: &MappedFile,
        cached: impl Fn(usize) -> bool,
    ) {
        let Some(target) = self.predictive.get(&layer) else {
            return;
        };
        let mut stats = self.stats.borrow_mut();
        stats.pending = None;
        let mut scores = vec![0.; target.bias.len()];
        if target.router.logits(0, &hidden(), &mut scores).is_err() {
            return;
        }
        scores.iter_mut().for_each(|v| *v = sigmoid(*v));
        let Ok(ids) = route(&scores, &target.bias, target.count) else {
            return;
        };
        for expert in ids.iter().copied().filter(|&id| !cached(id)) {
            for range in target.stack.ranges(expert) {
                let len = range.len();
                match mapping.prefetch(range) {
                    Ok(()) => stats.advice_bytes += len as u64,
                    Err(_) => stats.errors += 1,
                }
            }
        }
        stats.pending = Some((layer, ids.into_iter().map(|id| id as u16).collect()));
    }

    pub(super) fn observe(&self, layer: usize, selected: &[usize]) {
        let mut stats = self.stats.borrow_mut();
        if let Some((target, predicted)) = stats.pending.take()
            && target == layer
        {
            let actual: Vec<_> = selected.iter().map(|&id| id as u16).collect();
            stats.meter.add(&predicted, &actual);
            if stats.meter.samples().is_power_of_two() {
                let (hits, total) = stats.meter.counts();
                eprintln!(
                    "Inkling GGUF RAM predictions {}",
                    serde_json::json!({
                        "visits": stats.meter.samples(), "hits": hits, "selected": total,
                        "advice_bytes": stats.advice_bytes, "advice_errors": stats.errors
                    })
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn predicted_expert_selects_only_its_two_gguf_ranges() {
        let stack = Stack {
            layer: 7,
            parts: [11..41, 90..105],
            experts: 3,
        };
        assert_eq!(stack.bytes(), 45);
        assert_eq!(stack.ranges(0), [11..21, 90..95]);
        assert_eq!(stack.ranges(2), [31..41, 100..105]);
    }
}
