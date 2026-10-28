//! Eager M1 diagnostic executor. The caller verifies the pinned GGUF identity.
//! Physical F16 and the host D128 norm/RoPE bridge are not serving qualification.

mod profile_counters;
mod profile_tests;
mod reset_tests;
mod selected;

use super::*;
use crate::family::flint::{
    cuda_kv::{KvAppend, LayerKv, LayerKvAllocation},
    gguf_weights,
    kv_plan::KvPlan,
};
use crate::weights::{GgufType, dequantize};

pub(in crate::family::flint::cuda_ffn) struct CudaDecode<'a> {
    ctx: &'a CudaContext,
    stream: &'a DeviceStream,
    file: &'a File,
    header: &'a GgufHeader,
    plan: KvPlan,
    kv: Vec<LayerKv>,
    poisoned: bool,
    selected: Option<selected::Selected<'a>>,
    fail_after_layer: Option<usize>,
}

impl<'a> CudaDecode<'a> {
    pub(in crate::family::flint::cuda_ffn) fn new(
        ctx: &'a CudaContext,
        stream: &'a DeviceStream,
        source: (&'a File, &'a GgufHeader),
        capacity: usize,
    ) -> Result<Self, EngError> {
        let plan = KvPlan::new(capacity, 1, false)?;
        let kv = (0..49)
            .map(|index| LayerKvAllocation::new(&plan, index)?.allocate(ctx))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            ctx,
            stream,
            file: source.0,
            header: source.1,
            plan,
            kv,
            poisoned: false,
            selected: None,
            fail_after_layer: None,
        })
    }

    pub(in crate::family::flint::cuda_ffn) fn new_selected(
        ctx: &'a CudaContext,
        stream: &'a DeviceStream,
        source: (&'a File, &'a GgufHeader),
        capacity: usize,
        hot_n: usize,
    ) -> Result<Self, EngError> {
        selected::validate_request(hot_n, std::env::var("FLINT_CUDA_FFN_MMQ"))?;
        let mut decode = Self::new(ctx, stream, source, capacity)?;
        let kv_bytes = decode
            .kv
            .iter()
            .map(|kv| kv.keys.bytes() + kv.values.bytes())
            .sum();
        decode.selected = Some(selected::Selected::new(
            ctx,
            source,
            &decode.plan,
            kv_bytes,
        )?);
        Ok(decode)
    }

    pub(in crate::family::flint::cuda_ffn) fn fetch_stats(&self) -> &selected::FetchStats {
        &self
            .selected
            .as_ref()
            .expect("selected decode constructor")
            .stats
    }

    pub(in crate::family::flint::cuda_ffn) fn prepared_enabled(&self) -> bool {
        self.selected
            .as_ref()
            .is_some_and(|selected| selected.source.prepared_enabled())
    }

    pub(in crate::family::flint::cuda_ffn) fn tokens(&self) -> usize {
        self.kv[0].cursor.committed_prefix()
    }

    pub(in crate::family::flint::cuda_ffn) fn reset(&mut self) -> Result<(), EngError> {
        self.stream.synchronize()?;
        for cache in &mut self.kv {
            cache.cursor.reset_after_quiescence();
        }
        self.poisoned = false;
        Ok(())
    }

    pub(in crate::family::flint::cuda_ffn) fn step(&mut self, token: u32) -> Result<u32, EngError> {
        if token >= 150144 {
            return Err(bad("decode token is outside the vocabulary"));
        }
        if self.poisoned {
            return Err(EngError::Backend {
                operation: "flint_decode",
                message: "partial decode step requires a quiescent reset".into(),
            });
        }
        if self.tokens() == self.plan.context_rows() {
            return Err(EngError::ContextLimit(
                "Flint decode context is full".into(),
            ));
        }
        self.poisoned = true;
        let result = self.compute(token);
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    fn compute(&mut self, token: u32) -> Result<u32, EngError> {
        let position = self.tokens();
        if let Some(selected) = &mut self.selected {
            selected.stats.cold_experts.clear();
            selected.stats.payload_bytes = 0;
        }
        let row = gguf_weights::embedding_row(self.header, token)?;
        let bytes = gguf::raw(self.file, row).map_err(|e| bad(&e))?;
        let embedding = dequantize(GgufType::Bf16, &bytes)?;
        let mut state = self.ctx.upload(&f32_bytes(&embedding))?;
        let mut appends = Vec::with_capacity(49);
        for (index, cache) in self.kv.iter_mut().enumerate() {
            let append = cache.cursor.begin_append(position, 1)?;
            let eager;
            let (layer, fetch) = if let Some(selected) = &mut self.selected {
                (
                    &selected.layers[index],
                    Some(selected::Fetch {
                        layer: index as u32,
                        source: &mut selected.source,
                        banks: &mut selected.banks,
                        stats: &mut selected.stats,
                    }),
                )
            } else {
                // shortcut: old diagnostic restages all weights; retain as the continuation oracle.
                eager = weights::Layer::load(
                    self.ctx,
                    self.file,
                    self.header,
                    index as u32,
                    self.plan.cache_rows(index)?,
                )?;
                (&eager, None)
            };
            state = layer_step(
                self.ctx,
                self.stream,
                &state,
                layer,
                [&cache.keys, &cache.values],
                &append,
                fetch,
            )?;
            if self.fail_after_layer == Some(index) {
                return Err(EngError::Checkpoint(format!(
                    "injected after layer {index} KV write"
                )));
            }
            appends.push(append);
        }
        let logits = {
            let _head = super::super::profile::wall("decode_head");
            super::model::head(
                self.ctx,
                self.stream,
                (self.file, self.header),
                &[state],
                false,
            )?
        };
        if logits.len() != 150144 || logits.iter().any(|v| !v.is_finite()) {
            return Err(bad(
                "decode head has invalid dimensions or nonfinite logits",
            ));
        }
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
            .ok_or_else(|| bad("decode head is empty"))?
            .0 as u32;
        self.stream.synchronize()?;
        // No layer prefix becomes readable until the entire step and head have completed.
        for append in appends {
            append.commit();
        }
        Ok(token)
    }
}

fn layer_step(
    ctx: &CudaContext,
    stream: &DeviceStream,
    input: &DeviceBuffer,
    layer: &weights::Layer,
    cache: [&DeviceBuffer; 2],
    append: &KvAppend<'_>,
    fetch: Option<selected::Fetch<'_, '_>>,
) -> Result<DeviceBuffer, EngError> {
    let norm = normalized(ctx, stream, input, &layer.pre_norm)?;
    let q = projection(ctx, stream, &norm, &layer.qkv[0])?;
    let k = projection(ctx, stream, &norm, &layer.qkv[1])?;
    let v = projection(ctx, stream, &norm, &layer.qkv[2])?;
    let attended = attention(ctx, stream, [&q, &k, &v], layer, cache, append)?;
    let projected = projection(ctx, stream, &attended, &layer.output)?;
    let residual_input = residual(ctx, stream, input, &projected)?;
    let norm = normalized(ctx, stream, &residual_input, &layer.post_norm)?;
    let ffn = if let Some(fetch) = fetch {
        fetch.forward(ctx, stream, layer, &norm)?
    } else {
        layer.ffn(ctx, stream, &norm)?
    };
    residual(ctx, stream, &residual_input, &ffn)
}

fn attention(
    ctx: &CudaContext,
    stream: &DeviceStream,
    qkv: [&DeviceBuffer; 3],
    layer: &weights::Layer,
    cache: [&DeviceBuffer; 2],
    append: &KvAppend<'_>,
) -> Result<DeviceBuffer, EngError> {
    let position = append.live_range().end - 1;
    let slot = append.slot(position)?;
    let bridge = super::super::profile::wall("decode_host_qk_write");
    let q = values(stream, qkv[0])?;
    let k = values(stream, qkv[1])?;
    let [q, k] = real::qk_producer_oracle(
        &q,
        &k,
        [&layer.qk_norm[0], &layer.qk_norm[1]],
        layer.sliding.then_some(position as u32),
    )
    .map_err(|e| bad(&e))?;
    let v = values(stream, qkv[2])?;
    for (plane, row) in cache.iter().zip([&k, &v]) {
        let row = real::half_cache(row).map_err(|e| bad(&e))?;
        let bytes = row
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_ne_bytes())
            .collect::<Vec<_>>();
        plane.write_at(stream, slot * 512 * 2, &bytes)?;
    }
    let q = ctx.upload(&f32_bytes(&q))?;
    drop(bridge);
    let geometry = CpfAttnGeometry {
        m: 1,
        n_head: 32,
        n_kv_head: 4,
        head_dim: 128,
    };
    let span = CpfAttnSpan {
        window: if layer.sliding { 513 } else { 0 },
        ring: 0,
    };
    let plan = cpf_attn_launch_plan(
        geometry,
        position + 1,
        span.window as usize,
        CPF_ATTN_SM_COUNT,
    )?;
    let part = ctx.allocate(cpf_attn_part_bytes(geometry, plan)?)?;
    let sinks = ctx.upload(&f32_bytes(&layer.sinks))?;
    let shape = RowShape {
        rows: 1,
        columns: 4096,
    };
    let mut frame = Frame::new(ctx, stream)?;
    let output = frame.allocate(shape)?;
    ctx.attn_prefill(
        stream,
        AttnPrefillCall {
            storage: AttnKvStorage::F16,
            kv_views: None,
            shape: AttnPrefillShape {
                m: 1,
                n_head: 32,
                n_kv_head: 4,
                head_dim: 128,
                cache_rows: layer.cache_rows,
                ld: 0,
                p0: position as i64,
            },
            span,
            split: CpfAttnSplit {
                max_splits: plan.max_splits,
                min_keys: plan.min_keys,
            },
            q_tile: plan.q_tile,
            scale: (128.0_f32).sqrt().recip(),
            q: q.range(0, q.bytes())?,
            proj: None,
            k_cache: cache[0].range(0, cache[0].bytes())?,
            v_cache: cache[1].range(0, cache[1].bytes())?,
            part: part.range(0, part.bytes())?,
            out: frame.range(output)?,
            status: Some(frame.flag()?),
            sinks: Some(sinks.range(0, sinks.bytes())?),
        },
    )?;
    let published = frame.publish(output, shape)?;
    frame.finish(published)
}
