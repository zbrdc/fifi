//! Per-visit state and streaming attention for the native Inkling CPU check.

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use crate::family::inkling2::switches::Switch;
use std::{collections::VecDeque, sync::Arc};

use super::cpu_forward::{CpuForward, dot, rms_norm, sigmoid};
use super::cpu_weights::Projection;
use super::{Attention, LayerVisit, phase_profile::Work};
use crate::EngError;

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use super::{cpu_weights::Weights, cuda_projection::gguf::attention};
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use crate::kernels::cuda::resident::inkling_local_prep::PrepShape;

pub(super) type AttentionProjection = [Vec<f32>; 8];

#[derive(Clone, Default)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct CausalConv(VecDeque<Vec<f32>>);

impl CausalConv {
    fn apply(
        &mut self,
        input: Vec<f32>,
        kernel: &[f32],
        taps: usize,
        dynamic: Option<DynamicTaps>,
    ) -> Vec<f32> {
        let kernels = if dynamic.is_some() { 2 } else { 1 };
        let norm = ((kernels * taps + 1) as f32).sqrt().recip();
        let width = input.len();
        let mut output = vec![0.0; width];
        for (lag, row) in std::iter::once(&input).chain(&self.0).enumerate() {
            for (channel, (result, &x)) in output.iter_mut().zip(row).enumerate() {
                let w = if let Some(dynamic) = &dynamic {
                    kernel[lag * width + channel]
                        + dynamic.coefficients[(channel / dynamic.head_width) * taps + lag]
                } else {
                    kernel[channel * taps + taps - 1 - lag]
                };
                *result += w * x;
            }
        }
        for (result, x) in output.iter_mut().zip(&input) {
            *result = (*result + x) * norm;
        }
        self.remember(input, taps);
        output
    }

    /// Records a raw row as the newest of the at most `taps - 1` history rows.
    fn remember(&mut self, raw: Vec<f32>, taps: usize) {
        self.0.push_front(raw);
        self.0.truncate(taps - 1);
    }
}

struct DynamicTaps {
    coefficients: Vec<f32>,
    head_width: usize,
}

#[derive(Clone, Default)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct VisitConvs {
    k: CausalConv,
    v: CausalConv,
    attn: CausalConv,
    mlp: CausalConv,
}

#[derive(Clone, Default)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(super) struct VisitState {
    // KV rows are immutable after append, so prefix snapshots share their storage.
    keys: VecDeque<Arc<Vec<f32>>>,
    values: VecDeque<Arc<Vec<f32>>>,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    position: usize,
    conv: VisitConvs,
    /// Set while the K/V rows and the four convolution histories of this visit exist only in
    /// the device ring and the resident token arenas, so the deques above are empty. Every
    /// host-orchestrated consumer calls `CpuForward::materialize_resident` first.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    resident: bool,
}

/// The parts of one visit overwritten by one forward row.
pub(super) struct VisitUndo {
    attention: AttentionUndo,
    mlp: CausalConv,
}

pub(super) struct AttentionUndo {
    evicted: Option<[Arc<Vec<f32>>; 2]>,
    k: CausalConv,
    v: CausalConv,
    attn: CausalConv,
}

impl AttentionUndo {
    pub(super) fn with_mlp(self, state: &VisitState) -> VisitUndo {
        VisitUndo {
            attention: self,
            mlp: state.conv.mlp.clone(),
        }
    }
}

#[derive(Clone, Copy)]
struct Heads {
    query: usize,
    kv: usize,
    width: usize,
}

impl Heads {
    fn q_channels(self) -> usize {
        self.query * self.width
    }
    fn kv_channels(self) -> usize {
        self.kv * self.width
    }
}

struct PreparedAttention {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    relative: Vec<f32>,
    sinks: Vec<f32>,
    gates: Vec<f32>,
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
impl PreparedAttention {
    fn device_row(&self) -> attention::Row<'_> {
        attention::Row {
            query: &self.q,
            key: &self.k,
            value: &self.v,
            relative: &self.relative,
            sinks: &self.sinks,
            gates: &self.gates,
        }
    }
}

impl CpuForward {
    fn prepare_attention(
        &self,
        attn: &str,
        heads: Heads,
        state: &mut VisitState,
        projected: AttentionProjection,
    ) -> Result<PreparedAttention, EngError> {
        self.prepare_attention_row(attn, heads, state, projected, false)
    }

    fn prepare_attention_row(
        &self,
        attn: &str,
        heads: Heads,
        state: &mut VisitState,
        projected: AttentionProjection,
        resident_query: bool,
    ) -> Result<PreparedAttention, EngError> {
        let [mut q, k, v, mut relative, sinks, gates, k_taps, v_taps] = projected;
        if !resident_query {
            self.scale_channels(&mut q, &format!("{attn}.q_l2_clip.scale"))?;
            clip_rows(&mut q, heads.width, self.math.qk_l2_clip_threshold);
        }
        let mut k = self.attention_plane(attn, "k", [k, k_taps], heads, &mut state.conv.k)?;
        let mut v = self.attention_plane(attn, "v", [v, v_taps], heads, &mut state.conv.v)?;
        clip_rows(&mut k, heads.width, self.math.qk_l2_clip_threshold);
        for head in v.chunks_exact_mut(heads.width) {
            rms_norm(head, self.math.rms_norm_eps);
        }
        if !resident_query {
            self.scale_channels(&mut relative, &format!("{attn}.rel_l2_clip.scale"))?;
            clip_rows(
                &mut relative,
                self.math.d_rel,
                self.math.rel_l2_clip_threshold,
            );
        }
        Ok(PreparedAttention {
            q,
            k,
            v,
            relative,
            sinks,
            gates,
        })
    }

    /// [`Self::prepare_attention_row`] without K and V, which the device prepares: their slots
    /// stay zero until the device's rows replace them.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    fn prepare_attention_query(
        &self,
        attn: &str,
        heads: Heads,
        projected: AttentionProjection,
        resident_query: bool,
    ) -> Result<PreparedAttention, EngError> {
        let [mut q, _, _, mut relative, sinks, gates, _, _] = projected;
        if !resident_query {
            self.scale_channels(&mut q, &format!("{attn}.q_l2_clip.scale"))?;
            clip_rows(&mut q, heads.width, self.math.qk_l2_clip_threshold);
            self.scale_channels(&mut relative, &format!("{attn}.rel_l2_clip.scale"))?;
            clip_rows(
                &mut relative,
                self.math.d_rel,
                self.math.rel_l2_clip_threshold,
            );
        }
        let slot = vec![0.0; heads.kv_channels()];
        Ok(PreparedAttention {
            q,
            k: slot.clone(),
            v: slot,
            relative,
            sinks,
            gates,
        })
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn local_decode_prep<'a>(
        &self,
        state: &'a VisitState,
        read_weights: &'a attention::LocalReadWeights<'a>,
    ) -> attention::LocalPrep<'a> {
        use crate::kernels::cuda::resident::inkling_local_prep::PrepMath;
        debug_assert_eq!(self.math.sconv_kernel_size, PrepShape::TAPS);
        attention::LocalPrep {
            math: PrepMath {
                qk_limit: self.math.qk_l2_clip_threshold,
                rel_limit: self.math.rel_l2_clip_threshold,
                eps: self.math.rms_norm_eps,
            },
            read_weights,
            raw: [&state.conv.k.0, &state.conv.v.0],
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn finish_local_decode(
        &self,
        layer: &str,
        state: &mut VisitState,
        result: attention::LocalResult,
    ) -> Result<Vec<f32>, EngError> {
        let attention::LocalResult {
            output,
            projected,
            rows: [raw_k, raw_v, k, v],
        } = result;
        let past = self.math.sconv_kernel_size - 1;
        for (history, raw) in [(&mut state.conv.k.0, raw_k), (&mut state.conv.v.0, raw_v)] {
            history.push_front(raw);
            history.truncate(past);
        }
        state.push(k, v, self.kv_capacity(Attention::Local));
        self.finish_attention(
            layer,
            state,
            &format!("{layer}.attn.wo_ud.weight"),
            output,
            projected,
        )
    }

    pub(super) fn attention(
        &self,
        layer: &str,
        visit: LayerVisit,
        state: &mut VisitState,
        projected: AttentionProjection,
    ) -> Result<Vec<f32>, EngError> {
        let heads = self.heads(visit.attention);
        let attn = format!("{layer}.attn");
        let prepared = self.prepare_attention(&attn, heads, state, projected)?;
        let output_name = format!("{attn}.wo_ud.weight");
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        let device_output = if let Weights::Gguf(weights) = &self.weights
            && let Some((context, attention)) = weights.resident_attention()
        {
            let _span = self.phase_span(Work::DeviceAttention);
            let projection = weights
                .relative_attention_projection(&output_name, [self.hidden(), prepared.q.len()])?;
            let projected = projection.is_some();
            let output = attention.step(
                context,
                visit.state_slot,
                prepared.device_row(),
                attention::History {
                    position: state.position,
                    keys: &state.keys,
                    values: &state.values,
                },
                projection,
            )?;
            Some((output, projected))
        } else {
            None
        };
        #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
        let device_output = None;
        let PreparedAttention {
            q,
            k,
            v,
            relative,
            sinks,
            gates,
        } = prepared;
        state.push(k, v, self.kv_capacity(visit.attention));

        let (out, projected_output) = if let Some(out) = device_output {
            out
        } else {
            let _span = self.phase_span(Work::CpuAttention);
            let table = self.weights.vec(
                &format!("{attn}.rel_logits_proj.proj"),
                self.math.d_rel * self.math.rel_extent,
            )?;
            let mut out = self.attend(state, heads, &q, &relative, &table, &sinks);
            for (out, gate) in out.iter_mut().zip(gates) {
                *out *= sigmoid(gate);
            }
            (out, false)
        };
        self.finish_attention(layer, state, &output_name, out, projected_output)
    }

    fn finish_attention(
        &self,
        layer: &str,
        state: &mut VisitState,
        output_name: &str,
        out: Vec<f32>,
        projected_output: bool,
    ) -> Result<Vec<f32>, EngError> {
        let raw = if projected_output {
            out
        } else {
            let _span = self.phase_span(Work::Projection);
            self.weights.project(output_name, &out, self.hidden())?
        };
        self.output_conv(
            raw,
            &format!("{layer}.attn_sconv.weight"),
            &mut state.conv.attn,
        )
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn attention_prefill_batch(
        &self,
        layer: &str,
        visit: LayerVisit,
        state: &mut VisitState,
        projections: Vec<AttentionProjection>,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let Weights::Gguf(weights) = &self.weights else {
            return Err(super::bad("Inkling attention batch requires GGUF"));
        };
        let (context, device) = weights
            .resident_attention()
            .ok_or_else(|| super::bad("Inkling attention batch requires resident state"))?;
        let heads = self.heads(visit.attention);
        let prefix = format!("{layer}.attn");
        let output_name = format!("{prefix}.wo_ud.weight");
        let projection = weights
            .relative_attention_projection(&output_name, [self.hidden(), heads.q_channels()])?;
        let capacity = device.prefill_capacity(context, visit.state_slot, projection.as_ref())?;
        if capacity < 2 {
            return projections
                .into_iter()
                .map(|p| self.attention(layer, visit, state, p))
                .collect();
        }
        let mut outputs = Vec::with_capacity(projections.len());
        let mut projections = projections.into_iter();
        while projections.len() != 0 {
            let tile = projections
                .by_ref()
                .take(capacity)
                .map(|p| self.prepare_attention(&prefix, heads, state, p))
                .collect::<Result<Vec<_>, _>>()?;
            let rows: Vec<_> = tile.iter().map(PreparedAttention::device_row).collect();
            let values = {
                let _span = self.phase_span(Work::DeviceAttention);
                device.step_prefill(
                    context,
                    visit.state_slot,
                    (&rows, None),
                    attention::History {
                        position: state.position,
                        keys: &state.keys,
                        values: &state.values,
                    },
                    projection.as_ref(),
                )?
            };
            for (prepared, out) in tile.into_iter().zip(values) {
                state.push(prepared.k, prepared.v, self.kv_capacity(visit.attention));
                outputs.push(self.finish_attention(
                    layer,
                    state,
                    &output_name,
                    out,
                    projection.is_some(),
                )?);
            }
        }
        Ok(outputs)
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn attention_prefill_bridge(
        &self,
        layer: &str,
        visit: LayerVisit,
        state: &mut VisitState,
        inputs: &[[Vec<f32>; 3]],
        mut resident: Option<&mut attention::prefill::PrefillConv>,
    ) -> Result<Option<Vec<Vec<f32>>>, EngError> {
        let resident_query = attention::prefill::resident_query_enabled();
        if (!resident_query && !attention::prefill::device_side_enabled()) || inputs.len() < 2 {
            return Ok(None);
        }
        let Weights::Gguf(weights) = &self.weights else {
            return Ok(None);
        };
        let widths = inputs[0].each_ref().map(Vec::len);
        let Some(projector) = self.attention_requests(layer, visit.attention, |calls| {
            weights.prefill_projection(widths, calls)
        })?
        else {
            return Ok(None);
        };
        let Some(bridge) = self.prefill_bridge(layer, visit, resident_query)? else {
            return Ok(None);
        };
        let keep = bridge.keep();
        let capacity = bridge
            .capacity
            .min(projector.capacity(inputs[0].each_ref().map(Vec::as_slice), keep)?);
        if capacity < 2 {
            return Ok(None);
        }
        // Without a caller's plane, the output convolution still runs on the device and the
        // convolved rows come back in place of the raw ones.
        let mut own = match resident {
            Some(_) => None,
            None => bridge.admit_conv(self, state, inputs.len())?,
        };
        let mut outputs = Vec::with_capacity(inputs.len());
        for tile in inputs.chunks(capacity) {
            let refs: Vec<_> = tile
                .iter()
                .map(|row| row.each_ref().map(Vec::as_slice))
                .collect();
            let (host, range) = projector.tile(&refs, keep)?;
            let conv = match (resident.as_deref_mut(), own.as_mut()) {
                (Some(conv), _) => Some((conv, false)),
                (None, Some(conv)) => Some((conv, true)),
                (None, None) => None,
            };
            outputs.extend(bridge.consume(self, state, host, range, conv)?);
        }
        if let Some(conv) = own {
            bridge.publish_conv_tail(state, conv)?;
        }
        if resident_query {
            note_resident_queries(false);
        } else {
            static USED: std::sync::Once = std::sync::Once::new();
            USED.call_once(|| {
                eprintln!("Inkling CUDA kept prefill attention sinks and gates resident")
            });
        }
        Ok(Some(outputs))
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn prefill_bridge(
        &self,
        layer: &str,
        visit: LayerVisit,
        resident_query: bool,
    ) -> Result<Option<PrefillBridge<'_>>, EngError> {
        let Weights::Gguf(weights) = &self.weights else {
            return Ok(None);
        };
        let Some((context, device)) = weights.resident_attention() else {
            return Ok(None);
        };
        let heads = self.heads(visit.attention);
        let prefix = format!("{layer}.attn");
        let output_name = format!("{prefix}.wo_ud.weight");
        // The device convolves, clips and normalizes K and V of whole tiles.
        let kv = Switch::PrefillKvPrep.on()
            && self.math.sconv_kernel_size == PrepShape::TAPS
            && heads.width == PrepShape::WIDTH;
        // Without a resident output projection, host fallback can reuse the logits scratch.
        let Some(projection) = weights
            .relative_attention_projection(&output_name, [self.hidden(), heads.q_channels()])?
        else {
            return Ok(None);
        };
        let (capacity, stride) = device.prefill_tile(
            context,
            visit.state_slot,
            Some(&projection),
            attention::prefill::projection_side(resident_query, kv),
        )?;
        if capacity < 2 {
            return Ok(None);
        }
        let scales = resident_query
            .then(|| self.prefill_query_scales(&prefix, heads))
            .transpose()?;
        Ok(Some(PrefillBridge {
            context,
            device,
            heads,
            visit,
            prefix,
            layer: layer.to_owned(),
            output_name,
            projection,
            capacity,
            stride,
            scales,
            kv,
        }))
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    fn prefill_query_scales(
        &self,
        prefix: &str,
        heads: Heads,
    ) -> Result<[std::borrow::Cow<'_, [f32]>; 2], EngError> {
        Ok([
            self.weights
                .vec(&format!("{prefix}.q_l2_clip.scale"), heads.q_channels())?,
            self.weights.vec(
                &format!("{prefix}.rel_l2_clip.scale"),
                heads.query * self.math.d_rel,
            )?,
        ])
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    fn prefill_query_clip<'a>(
        &self,
        scales: &'a [std::borrow::Cow<'_, [f32]>; 2],
    ) -> attention::prefill::QueryClip<'a> {
        attention::prefill::QueryClip {
            scales: scales.each_ref().map(AsRef::as_ref),
            limits: [
                self.math.qk_l2_clip_threshold,
                self.math.rel_l2_clip_threshold,
            ],
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn invalidate_device_attention(&self) {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        self.pass_tails_stale();
        if let Weights::Gguf(weights) = &self.weights
            && let Some((_, attention)) = weights.resident_attention()
        {
            attention.invalidate();
        }
    }

    pub(super) fn kv_capacity(&self, attention: Attention) -> usize {
        match attention {
            Attention::Local => self.spec.sliding_window_size as usize,
            Attention::Global => self.context,
        }
    }

    pub(super) fn project_attention(
        &self,
        layer: &str,
        attention: Attention,
        inputs: &[Vec<f32>; 3],
    ) -> Result<AttentionProjection, EngError> {
        self.attention_requests(layer, attention, |calls| {
            self.weights
                .attention_projections(inputs.each_ref().map(Vec::as_slice), calls)
        })
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn prefill_attention_projections(
        &self,
        weights: &super::gguf_weights::GgufTextWeights,
        layer: &str,
        attention: Attention,
        inputs: &[[Vec<f32>; 3]],
        allow_q8_row_reuse: bool,
    ) -> Result<Option<Vec<AttentionProjection>>, EngError> {
        let refs: Vec<_> = inputs
            .iter()
            .map(|input| input.each_ref().map(Vec::as_slice))
            .collect();
        self.attention_requests(layer, attention, |calls| {
            weights.attention_projections_rows(&refs, calls, allow_q8_row_reuse)
        })
    }

    pub(super) fn attention_requests<R>(
        &self,
        layer: &str,
        attention: Attention,
        run: impl FnOnce([Option<Projection<'_>>; 8]) -> R,
    ) -> R {
        let heads = self.heads(attention);
        let dynamic = attention == Attention::Global;
        let taps = self.math.sconv_kernel_size * heads.kv;
        let requests = [
            ("wq_du.weight", 0, heads.q_channels()),
            ("wk_dv.weight", 1, heads.kv_channels()),
            ("wv_dv.weight", 2, heads.kv_channels()),
            ("wr_du.weight", 0, heads.query * self.math.d_rel),
            ("attn_sink_proj.weight", 0, heads.query),
            ("gate_attn.weight", 0, heads.q_channels()),
            ("k_sconv.weight_proj", 1, taps),
            ("v_sconv.weight_proj", 2, taps),
        ];
        let names = requests.map(|(suffix, _, _)| format!("{layer}.attn.{suffix}"));
        let calls = std::array::from_fn(|i| {
            let (_, input, rows) = requests[i];
            (i < 6 || dynamic).then_some(Projection {
                name: &names[i],
                input,
                rows,
            })
        });
        let _span = self.phase_span(Work::Projection);
        run(calls)
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn local_query_channels(&self) -> usize {
        self.heads(Attention::Local).q_channels()
    }

    fn heads(&self, attention: Attention) -> Heads {
        let spec = &self.spec;
        let (query, kv, width) = match attention {
            Attention::Local => (
                spec.swa_num_attention_heads,
                spec.swa_num_key_value_heads,
                spec.swa_head_dim,
            ),
            Attention::Global => (
                spec.num_attention_heads,
                spec.num_key_value_heads,
                spec.head_dim,
            ),
        };
        Heads {
            query: query as usize,
            kv: kv as usize,
            width: width as usize,
        }
    }

    fn scale_channels(&self, x: &mut [f32], tensor: &str) -> Result<(), EngError> {
        let weights = self.weights.vec(tensor, x.len())?;
        for (x, weight) in x.iter_mut().zip(weights.iter()) {
            *x *= weight;
        }
        Ok(())
    }

    fn attention_plane(
        &self,
        attn: &str,
        plane: &str,
        [raw, coefficients]: [Vec<f32>; 2],
        heads: Heads,
        conv: &mut CausalConv,
    ) -> Result<Vec<f32>, EngError> {
        let taps = self.math.sconv_kernel_size;
        let _span = self.phase_span(Work::Convolution);
        let channels = heads.kv_channels();
        let (name, dynamic) = if coefficients.is_empty() {
            ("weight", None)
        } else {
            (
                "static_w",
                Some(DynamicTaps {
                    coefficients,
                    head_width: heads.width,
                }),
            )
        };
        let kernel = self
            .weights
            .vec(&format!("{attn}.{plane}_sconv.{name}"), taps * channels)?;
        Ok(conv.apply(raw, &kernel, taps, dynamic))
    }

    fn attend(
        &self,
        state: &VisitState,
        heads: Heads,
        q: &[f32],
        relative: &[f32],
        table: &[f32],
        sinks: &[f32],
    ) -> Vec<f32> {
        let mut output = vec![0.0; heads.q_channels()];
        let mut scores = vec![0.0; state.keys.len()];
        let scale = (heads.width as f32).sqrt().recip();
        let dims = self.math.d_rel;
        let extent = self.math.rel_extent;
        for (head, dst) in output.chunks_exact_mut(heads.width).enumerate() {
            let query = &q[head * heads.width..][..heads.width];
            let rel = &relative[head * dims..][..dims];
            let kv_offset = head / (heads.query / heads.kv) * heads.width;
            for (distance, (k, score)) in state.keys.iter().zip(&mut scores).enumerate() {
                let bias = if distance < extent {
                    table
                        .chunks_exact(extent)
                        .zip(rel)
                        .map(|(table, coefficient)| table[distance] * coefficient)
                        .sum()
                } else {
                    0.0
                };
                *score = scale * dot(query, &k[kv_offset..][..heads.width]) + bias;
            }
            let peak = scores.iter().copied().fold(sinks[head], f32::max);
            let mut sum = (sinks[head] - peak).exp();
            for (value, score) in state.values.iter().zip(&scores) {
                let mass = (score - peak).exp();
                sum += mass;
                for (out, v) in dst.iter_mut().zip(&value[kv_offset..][..heads.width]) {
                    *out += mass * v;
                }
            }
            for out in dst {
                *out /= sum;
            }
        }
        output
    }

    fn output_conv(
        &self,
        input: Vec<f32>,
        name: &str,
        state: &mut CausalConv,
    ) -> Result<Vec<f32>, EngError> {
        let taps = self.math.sconv_kernel_size;
        let _span = self.phase_span(Work::Convolution);
        let kernel = self.weights.vec(name, input.len() * taps)?;
        Ok(state.apply(input, &kernel, taps, None))
    }

    pub(super) fn mlp_conv(
        &self,
        input: Vec<f32>,
        layer: &str,
        state: &mut VisitState,
    ) -> Result<Vec<f32>, EngError> {
        self.output_conv(
            input,
            &format!("{layer}.mlp_sconv.weight"),
            &mut state.conv.mlp,
        )
    }
}

impl VisitState {
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn position(&self) -> usize {
        self.position
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn is_resident(&self) -> bool {
        self.resident
    }

    /// The device committed the row at `position`; the host keeps the cursor only.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn advance_resident(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.conv = VisitConvs::default();
        self.position += 1;
        self.resident = true;
    }

    /// Adopts rows read back from the device. `keys` and `values` run oldest last, as the
    /// ring yields them; each history plane runs newest first, as its arena holds it (the
    /// device keeps the raw samples, newest in row 0). The deques are newest first.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn adopt_resident(
        &mut self,
        (keys, values): (Vec<Vec<f32>>, Vec<Vec<f32>>),
        [k, v, attn, mlp]: [Vec<Vec<f32>>; 4],
        (past, capacity): (usize, usize),
    ) {
        debug_assert!(self.resident && self.keys.is_empty() && self.conv.k.0.is_empty());
        self.keys = keys.into_iter().rev().map(Arc::new).collect();
        self.values = values.into_iter().rev().map(Arc::new).collect();
        self.keys.truncate(capacity);
        self.values.truncate(capacity);
        for (history, rows) in [
            &mut self.conv.k.0,
            &mut self.conv.v.0,
            &mut self.conv.attn.0,
            &mut self.conv.mlp.0,
        ]
        .into_iter()
        .zip([k, v, attn, mlp])
        {
            *history = rows.into_iter().collect();
            history.truncate(past);
        }
        self.resident = false;
    }

    /// Adopts what a device-resident prefill pass read back (`INVARIANTS.md` of the pass lane,
    /// section 1): `keys` and `values` newest first, as the ring yields them reversed; the four
    /// raw convolution histories oldest first with three rows each, of which the last `valid`
    /// exist; and the cursor `position`.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn publish_pass(
        &mut self,
        (keys, values): (Vec<Vec<f32>>, Vec<Vec<f32>>),
        [k, v, attn, mlp]: [Vec<Vec<f32>>; 4],
        (valid, capacity, position): (usize, usize, usize),
    ) {
        self.keys = keys.into_iter().rev().map(Arc::new).collect();
        self.values = values.into_iter().rev().map(Arc::new).collect();
        self.keys.truncate(capacity);
        self.values.truncate(capacity);
        for (history, rows) in [
            &mut self.conv.k.0,
            &mut self.conv.v.0,
            &mut self.conv.attn.0,
            &mut self.conv.mlp.0,
        ]
        .into_iter()
        .zip([k, v, attn, mlp])
        {
            *history = rows.into_iter().rev().take(valid).collect();
            history.truncate(past_rows());
        }
        self.position = position;
        self.resident = false;
    }

    /// The four raw convolution histories, newest first, as the host keeps them.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn histories(&self) -> [Vec<Vec<f32>>; 4] {
        [
            self.conv.k.0.iter().cloned().collect(),
            self.conv.v.0.iter().cloned().collect(),
            self.conv.attn.0.iter().cloned().collect(),
            self.conv.mlp.0.iter().cloned().collect(),
        ]
    }

    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    pub(super) fn resident_history_identity(&self) -> Option<std::sync::Weak<Vec<f32>>> {
        self.keys.front().map(Arc::downgrade)
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn resident_conv_histories(&self) -> [&VecDeque<Vec<f32>>; 4] {
        [
            &self.conv.k.0,
            &self.conv.v.0,
            &self.conv.attn.0,
            &self.conv.mlp.0,
        ]
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn publish_resident_visit(
        &mut self,
        [raw_k, raw_v, k, v, attn, mlp]: [Vec<f32>; 6],
        (past, capacity): (usize, usize),
    ) {
        for (history, input) in [
            (&mut self.conv.k.0, raw_k),
            (&mut self.conv.v.0, raw_v),
            (&mut self.conv.attn.0, attn),
            (&mut self.conv.mlp.0, mlp),
        ] {
            history.push_front(input);
            history.truncate(past);
        }
        self.push(k, v, capacity);
    }

    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    pub(super) fn shadow_rows(&self) -> Vec<&[f32]> {
        self.keys
            .iter()
            .chain(&self.values)
            .map(|v| v.as_slice())
            .chain(
                self.resident_conv_histories()
                    .into_iter()
                    .flat_map(|rows| rows.iter().map(Vec::as_slice)),
            )
            .collect()
    }

    /// `shadow_rows` by plane: keys, values, then the four convolution histories.
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    pub(super) fn shadow_planes(&self) -> [Vec<&[f32]>; 6] {
        fn owned(rows: &VecDeque<Arc<Vec<f32>>>) -> Vec<&[f32]> {
            rows.iter().map(|row| row.as_slice()).collect()
        }
        fn conv(rows: &VecDeque<Vec<f32>>) -> Vec<&[f32]> {
            rows.iter().map(Vec::as_slice).collect()
        }
        [
            owned(&self.keys),
            owned(&self.values),
            conv(&self.conv.k.0),
            conv(&self.conv.v.0),
            conv(&self.conv.attn.0),
            conv(&self.conv.mlp.0),
        ]
    }

    /// Per-part fingerprints (`keys, values, k, v, attn, mlp`) and the full first and last K/V
    /// rows and history rows of the visit, for a parity run that must name the differing part.
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    pub(super) fn parity_parts(&self) -> (Vec<(&'static str, usize, u64)>, serde_json::Value) {
        fn plane<'a>(rows: impl IntoIterator<Item = &'a Vec<f32>>) -> (usize, u64) {
            use std::hash::{DefaultHasher, Hash, Hasher};
            let mut hash = DefaultHasher::new();
            let mut len = 0;
            for row in rows {
                len += 1;
                for &value in row {
                    value.to_bits().hash(&mut hash);
                }
            }
            (len, hash.finish())
        }
        let keys: Vec<_> = self.keys.iter().map(Arc::as_ref).collect();
        let values: Vec<_> = self.values.iter().map(Arc::as_ref).collect();
        let parts = vec![
            ("keys", plane(keys.iter().copied())),
            ("values", plane(values.iter().copied())),
            ("k", plane(self.conv.k.0.iter())),
            ("v", plane(self.conv.v.0.iter())),
            ("attn", plane(self.conv.attn.0.iter())),
            ("mlp", plane(self.conv.mlp.0.iter())),
        ]
        .into_iter()
        .map(|(name, (len, hash))| (name, len, hash))
        .collect();
        fn bits<'a>(rows: impl IntoIterator<Item = &'a Vec<f32>>) -> serde_json::Value {
            serde_json::json!(
                rows.into_iter()
                    .map(|row| row.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            )
        }
        fn firsts<'a>(rows: &[&'a Vec<f32>]) -> Vec<&'a Vec<f32>> {
            match rows {
                [] => Vec::new(),
                rows => vec![rows[0], rows[rows.len() - 1]],
            }
        }
        let detail = serde_json::json!({
            "position": self.position,
            "keys": bits(firsts(&keys).into_iter()),
            "values": bits(firsts(&values).into_iter()),
            "k": bits(self.conv.k.0.iter()),
            "v": bits(self.conv.v.0.iter()),
            "attn": bits(self.conv.attn.0.iter()),
            "mlp": bits(self.conv.mlp.0.iter()),
        });
        (parts, detail)
    }

    /// Every K/V row and history row of the visit, as bits (for a per-row parity comparison).
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    pub(super) fn parity_rows(&self) -> serde_json::Value {
        fn bits<'a>(rows: impl IntoIterator<Item = &'a Vec<f32>>) -> serde_json::Value {
            serde_json::json!(
                rows.into_iter()
                    .map(|row| row.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            )
        }
        serde_json::json!({
            "keys": bits(self.keys.iter().map(Arc::as_ref)),
            "values": bits(self.values.iter().map(Arc::as_ref)),
            "k": bits(self.conv.k.0.iter()),
            "v": bits(self.conv.v.0.iter()),
            "attn": bits(self.conv.attn.0.iter()),
            "mlp": bits(self.conv.mlp.0.iter()),
        })
    }

    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    pub(super) fn parity_fingerprint(&self) -> u64 {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let mut hash = DefaultHasher::new();
        self.position.hash(&mut hash);
        fn plane<'a>(rows: impl ExactSizeIterator<Item = &'a Vec<f32>>, hash: &mut DefaultHasher) {
            rows.len().hash(hash);
            for row in rows {
                row.len().hash(hash);
                for &value in row {
                    value.to_bits().hash(hash);
                }
            }
        }
        plane(self.keys.iter().map(Arc::as_ref), &mut hash);
        plane(self.values.iter().map(Arc::as_ref), &mut hash);
        for conv in [&self.conv.k, &self.conv.v, &self.conv.attn, &self.conv.mlp] {
            plane(conv.0.iter(), &mut hash);
        }
        hash.finish()
    }

    pub(super) fn checkpoint(&self, capacity: usize) -> VisitUndo {
        self.checkpoint_attention(capacity).with_mlp(self)
    }

    pub(super) fn checkpoint_attention(&self, capacity: usize) -> AttentionUndo {
        let evicted = if self.keys.len() == capacity {
            self.keys
                .back()
                .zip(self.values.back())
                .map(|(key, value)| [Arc::clone(key), Arc::clone(value)])
        } else {
            None
        };
        AttentionUndo {
            evicted,
            k: self.conv.k.clone(),
            v: self.conv.v.clone(),
            attn: self.conv.attn.clone(),
        }
    }

    pub(super) fn rewind(&mut self, undo: VisitUndo) {
        let AttentionUndo {
            evicted,
            k,
            v,
            attn,
        } = undo.attention;
        self.keys.pop_front();
        self.values.pop_front();
        if let Some([key, value]) = evicted {
            self.keys.push_back(key);
            self.values.push_back(value);
        }
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        {
            self.position -= 1;
        }
        // A resident state has no host rows to undo; its first host consumer materializes.
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        debug_assert!(!self.resident);
        self.conv = VisitConvs {
            k,
            v,
            attn,
            mlp: undo.mlp,
        };
    }

    fn push(&mut self, key: Vec<f32>, value: Vec<f32>, length: usize) {
        self.keys.push_front(Arc::new(key));
        self.values.push_front(Arc::new(value));
        self.keys.truncate(length);
        self.values.truncate(length);
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        {
            self.position += 1;
            self.resident = false;
        }
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(super) struct PrefillBridge<'a> {
    context: &'a crate::kernels::cuda::CudaContext,
    device: &'a attention::ResidentAttention,
    heads: Heads,
    visit: LayerVisit,
    prefix: String,
    layer: String,
    output_name: String,
    projection: attention::OutputProjection<'a>,
    capacity: usize,
    stride: usize,
    scales: Option<[std::borrow::Cow<'a, [f32]>; 2]>,
    kv: bool,
}

/// The MLP output convolution of a prompt visit on the device: the experts write their raw rows
/// into the admitted plane, one launch convolves every row, and the raw tail becomes the visit's
/// history.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
impl CpuForward {
    pub(super) fn admit_prompt_mlp_conv(
        &self,
        state: &VisitState,
        layer: &str,
        rows: usize,
    ) -> Result<Option<attention::prefill::PrefillConv>, EngError> {
        if !Switch::PrefillResidentExpertConv.on() || self.math.sconv_kernel_size != 4 || rows < 2 {
            return Ok(None);
        }
        let Weights::Gguf(weights) = &self.weights else {
            return Ok(None);
        };
        let Some((context, device)) = weights.resident_attention() else {
            return Ok(None);
        };
        let width = self.hidden();
        let filter = self
            .weights
            .vec(&format!("{layer}.mlp_sconv.weight"), width * 4)?;
        device.admit_prefill_conv(context, [rows, width], &state.conv.mlp.0, &filter)
    }

    /// Convolves the fully written expert plane, publishes its tail and returns the convolved
    /// rows in order.
    pub(super) fn finish_prompt_mlp_conv(
        &self,
        state: &mut VisitState,
        mut conv: attention::prefill::PrefillConv,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let _span = self.phase_span(Work::Convolution);
        let (context, device) = self.prompt_conv_device()?;
        let rows = device.finish_prefill_expert_conv_rows(context, &mut conv)?;
        state.conv.mlp.0 = device.prefill_conv_tail(conv)?;
        Ok(rows)
    }

    /// Publishes the raw tail of a plane whose convolved rows the caller already consumed.
    pub(super) fn publish_prompt_mlp_tail(
        &self,
        state: &mut VisitState,
        conv: attention::prefill::PrefillConv,
    ) -> Result<(), EngError> {
        state.conv.mlp.0 = self.prompt_conv_device()?.1.prefill_conv_tail(conv)?;
        Ok(())
    }

    fn prompt_conv_device(
        &self,
    ) -> Result<
        (
            &crate::kernels::cuda::CudaContext,
            &attention::ResidentAttention,
        ),
        EngError,
    > {
        match &self.weights {
            Weights::Gguf(weights) => weights.resident_attention(),
            _ => None,
        }
        .ok_or_else(|| super::bad("Inkling prompt convolution lost its CUDA attention"))
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
impl PrefillBridge<'_> {
    /// The projection outputs this bridge keeps on the device.
    pub(super) fn keep(&self) -> &'static [usize] {
        attention::prefill::projection_side(self.scales.is_some(), self.kv)
    }

    pub(super) fn admit_conv(
        &self,
        forward: &CpuForward,
        state: &VisitState,
        rows: usize,
    ) -> Result<Option<attention::prefill::PrefillConv>, EngError> {
        if !Switch::PrefillOutputConv.on() || forward.math.sconv_kernel_size != 4 {
            return Ok(None);
        }
        let width = forward.hidden();
        let filter = forward
            .weights
            .vec(&format!("{}.attn_sconv.weight", self.layer), width * 4)?;
        self.device
            .admit_prefill_conv(self.context, [rows, width], &state.conv.attn.0, &filter)
    }

    /// The device K/V preparation of one sub-tile: the visit's raw history and weights, and the
    /// dynamic taps of each row when the visit has them.
    fn kv_prep<'s>(
        &'s self,
        forward: &'s CpuForward,
        state: &'s VisitState,
        coefficients: Vec<f32>,
    ) -> Result<attention::prefill::KvPrep<'s>, EngError> {
        let heads = self.heads;
        let channels = heads.kv_channels();
        let taps = forward.math.sconv_kernel_size;
        // The host convolution's rule: per-row taps were projected exactly when they are dynamic.
        let dynamic = !coefficients.is_empty();
        let name = if dynamic { "static_w" } else { "weight" };
        let [k, v] = ["k", "v"].map(|plane| {
            forward.weights.vec(
                &format!("{}.{plane}_sconv.{name}", self.prefix),
                taps * channels,
            )
        });
        let coefficients = dynamic.then_some(coefficients);
        Ok(attention::prefill::KvPrep {
            kv_heads: heads.kv,
            history: [&state.conv.k.0, &state.conv.v.0],
            weights: [k?, v?],
            coefficients,
            limit: forward.math.qk_l2_clip_threshold,
            eps: forward.math.rms_norm_eps,
            prepared: std::cell::Cell::new(None),
        })
    }

    pub(super) fn publish_conv_tail(
        &self,
        state: &mut VisitState,
        conv: attention::prefill::PrefillConv,
    ) -> Result<(), EngError> {
        state.conv.attn.0 = self.device.prefill_conv_tail(conv)?;
        static USED: std::sync::Once = std::sync::Once::new();
        USED.call_once(|| {
            eprintln!("Inkling CUDA used prefill output convolution and published only raw tail")
        });
        Ok(())
    }

    pub(super) fn consume(
        &self,
        forward: &CpuForward,
        state: &mut VisitState,
        host: Vec<AttentionProjection>,
        range: crate::kernels::cuda::DeviceRange<'_>,
        mut conv: Option<(&mut attention::prefill::PrefillConv, bool)>,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let query = self
            .scales
            .as_ref()
            .map(|scales| forward.prefill_query_clip(scales));
        let stride = self.stride;
        let count = host.len();
        if count.checked_mul(stride) != Some(range.bytes()) {
            return Err(super::bad(
                "incorrect Inkling device prefill projection tile extent",
            ));
        }
        let mut host = host.into_iter();
        let mut output = Vec::with_capacity(count);
        // Keep the controller's original tile size. An attention sub-tile is
        // just a contiguous slice of the borrowed sidecar, with no new copy.
        for first in (0..count).step_by(self.capacity) {
            let count = (count - first).min(self.capacity);
            let sub = range
                .buffer()
                .range(range.offset() + first * stride, count * stride)?;
            output.extend(self.consume_rows(
                forward,
                state,
                host.by_ref().take(count),
                attention::prefill::Sidecar {
                    range: sub,
                    query,
                    kv: None,
                },
                conv.as_mut().map(|(conv, read)| (&mut **conv, *read)),
            )?);
        }
        Ok(output)
    }

    fn consume_rows(
        &self,
        forward: &CpuForward,
        state: &mut VisitState,
        host: impl Iterator<Item = AttentionProjection>,
        side: attention::prefill::Sidecar<'_>,
        conv: Option<(&mut attention::prefill::PrefillConv, bool)>,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let resident_query = side.query.is_some();
        let mut prepared = Vec::new();
        let mut coefficients = Vec::new();
        for p in host {
            prepared.push(if self.kv {
                coefficients.extend(p[6].iter().chain(&p[7]));
                forward.prepare_attention_query(&self.prefix, self.heads, p, resident_query)?
            } else {
                forward.prepare_attention_row(&self.prefix, self.heads, state, p, resident_query)?
            });
        }
        let kv = self
            .kv
            .then(|| self.kv_prep(forward, state, coefficients))
            .transpose()?;
        let side = attention::prefill::Sidecar {
            kv: kv.as_ref(),
            ..side
        };
        let rows: Vec<_> = prepared.iter().map(PreparedAttention::device_row).collect();
        let convolved = conv.is_some();
        let status_only = matches!(conv, Some((_, false)));
        let values = {
            let _span = forward.phase_span(Work::DeviceAttention);
            let history = attention::History {
                position: state.position,
                keys: &state.keys,
                values: &state.values,
            };
            // Finish consuming the sidecar before the next tile can overwrite it.
            if let Some(target) = conv {
                self.device.step_prefill_conv(
                    self.context,
                    self.visit.state_slot,
                    (&rows, Some(side)),
                    history,
                    &self.projection,
                    target,
                )?
            } else {
                self.device.step_prefill(
                    self.context,
                    self.visit.state_slot,
                    (&rows, Some(side)),
                    history,
                    Some(&self.projection),
                )?
            }
        };
        drop(rows);
        if let Some(kv) = kv {
            let device = kv
                .prepared
                .into_inner()
                .ok_or_else(|| super::bad("missing CUDA K/V preparation readback"))?;
            for (row, [k, v]) in prepared.iter_mut().zip(device.rows) {
                (row.k, row.v) = (k, v);
            }
            let taps = forward.math.sconv_kernel_size;
            for [k, v] in device.raw_tail {
                state.conv.k.remember(k, taps);
                state.conv.v.remember(v, taps);
            }
        }
        // The device ring is only a cache: publish the prepared host K/V
        // after the device status and output have passed validation.
        let capacity = forward.kv_capacity(self.visit.attention);
        if status_only {
            for row in prepared {
                state.push(row.k, row.v, capacity);
            }
            return Ok(values);
        }
        prepared
            .into_iter()
            .zip(values)
            .map(|(prepared, out)| {
                state.push(prepared.k, prepared.v, capacity);
                if convolved {
                    Ok(out)
                } else {
                    forward.finish_attention(&self.layer, state, &self.output_name, out, true)
                }
            })
            .collect()
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(super) fn note_resident_queries(chained: bool) {
    static PLAIN: std::sync::Once = std::sync::Once::new();
    static CHAINED: std::sync::Once = std::sync::Once::new();
    let (receipt, path) = if chained {
        (&CHAINED, "chained")
    } else {
        (&PLAIN, "plain")
    };
    receipt.call_once(|| {
        eprintln!(
            "Inkling CUDA kept {path} prefill attention Q, relative, sinks and gates resident"
        )
    });
}

/// Rows a convolution history keeps between groups (`taps - 1` of the four-tap kernels).
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(super) fn past_rows() -> usize {
    3
}

pub(super) fn clip_rows(rows: &mut [f32], width: usize, limit: f32) {
    for row in rows.chunks_exact_mut(width) {
        let scale = (limit / (dot(row, row).sqrt() + 1e-6)).min(1.0);
        for value in row {
            *value *= scale;
        }
    }
}

#[test]
fn causal_conv_uses_per_visit_history_and_native_tap_order() {
    let mut first = CausalConv::default();
    let mut second = CausalConv::default();
    let weights = [2.0, 3.0];
    let root = 3.0_f32.sqrt();
    assert!((first.apply(vec![4.0], &weights, 2, None)[0] - 16.0 / root).abs() < 1e-6);
    assert!((second.apply(vec![5.0], &weights, 2, None)[0] - 20.0 / root).abs() < 1e-6);
    assert!((first.apply(vec![5.0], &weights, 2, None)[0] - 28.0 / root).abs() < 1e-6);
    let dynamic = |coefficients| {
        Some(DynamicTaps {
            coefficients,
            head_width: 1,
        })
    };
    let mut full = CausalConv::default();
    full.apply(vec![4.0], &[3.0, 2.0], 2, dynamic(vec![1.0, 2.0]));
    let actual = full.apply(vec![5.0], &[3.0, 2.0], 2, dynamic(vec![1.0, 2.0]));
    assert!((actual[0] - 41.0 / 5.0_f32.sqrt()).abs() < 2e-6);
}

#[test]
fn sub_tiles_carry_the_history_one_tile_would_leave() {
    // The device returns each sub-tile's last raw rows, at most `taps - 1`, oldest first.
    let taps = 4;
    let rows: Vec<Vec<f32>> = (0..7).map(|i| vec![i as f32, -(i as f32)]).collect();
    for prior in 0..taps {
        let mut whole = CausalConv::default();
        for row in &rows[..prior] {
            whole.apply(row.clone(), &[0.5; 8], taps, None);
        }
        let start = whole.clone();
        for row in &rows[prior..] {
            whole.apply(row.clone(), &[0.5; 8], taps, None);
        }
        for split in [vec![7 - prior], vec![1, 6 - prior], vec![2, 1, 4 - prior]] {
            let mut carried = start.clone();
            let mut first = prior;
            for count in split {
                let tile = &rows[first..first + count];
                for raw in &tile[count - count.min(taps - 1)..] {
                    carried.remember(raw.clone(), taps);
                }
                first += count;
            }
            assert_eq!(carried, whole, "prior {prior}");
        }
    }
}

#[test]
fn rewind_restores_both_evicted_kv_planes_and_every_convolution() {
    fn advance(visit: &mut VisitState, value: f32, limit: usize) {
        let convs = &mut visit.conv;
        for (phase, conv) in [&mut convs.k, &mut convs.v, &mut convs.attn, &mut convs.mlp]
            .into_iter()
            .enumerate()
        {
            conv.apply(vec![value + phase as f32], &[0.1; 3], 3, None);
        }
        visit.push(vec![value], vec![-value], limit);
    }

    let mut visits = [VisitState::default(), VisitState::default()];
    let limits = [2, 5];
    for (index, (visit, limit)) in visits.iter_mut().zip(limits).enumerate() {
        for row in 0..3 {
            advance(visit, (index * 10 + row) as f32, limit);
        }
    }
    let committed = visits.clone();
    let mut journal = Vec::new();
    for row in 3..6 {
        let mut undo = Vec::new();
        for (index, (visit, limit)) in visits.iter_mut().zip(limits).enumerate() {
            undo.push(visit.checkpoint(limit));
            advance(visit, (index * 10 + row) as f32, limit);
        }
        journal.push(undo);
    }
    for undo in journal.into_iter().rev() {
        for (visit, undo) in visits.iter_mut().zip(undo) {
            visit.rewind(undo);
        }
    }
    assert_eq!(visits, committed);
}
