//! Per-visit state and streaming attention for the native Inkling CPU check.

use std::{collections::VecDeque, sync::Arc};

use super::Attention;
use super::cpu_forward::{CpuForward, dot, rms_norm, sigmoid};
use crate::EngError;

#[derive(Clone, Default)]
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
        self.0.push_front(input);
        self.0.truncate(taps - 1);
        output
    }
}

struct DynamicTaps {
    coefficients: Vec<f32>,
    head_width: usize,
}

#[derive(Clone, Default)]
pub(super) struct VisitState {
    // KV rows are immutable after append, so prefix snapshots share their storage.
    keys: VecDeque<Arc<Vec<f32>>>,
    values: VecDeque<Arc<Vec<f32>>>,
    k_conv: CausalConv,
    v_conv: CausalConv,
    attn_conv: CausalConv,
    mlp_conv: CausalConv,
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

impl CpuForward {
    pub(super) fn attention(
        &self,
        layer: &str,
        attention: Attention,
        inputs: &[Vec<f32>; 3],
        state: &mut VisitState,
    ) -> Result<Vec<f32>, EngError> {
        let heads = self.heads(attention);
        let attn = format!("{layer}.attn");
        let [q_in, k_in, v_in] = inputs;
        let mut q =
            self.weights
                .project(&format!("{attn}.wq_du.weight"), q_in, heads.q_channels())?;
        self.scale_channels(&mut q, &format!("{attn}.q_l2_clip.scale"))?;
        clip_rows(&mut q, heads.width, self.math.qk_l2_clip_threshold);

        let mut k = self.attention_plane(&attn, "k", k_in, heads, attention, &mut state.k_conv)?;
        let mut v = self.attention_plane(&attn, "v", v_in, heads, attention, &mut state.v_conv)?;
        clip_rows(&mut k, heads.width, self.math.qk_l2_clip_threshold);
        for head in v.chunks_exact_mut(heads.width) {
            rms_norm(head, self.math.rms_norm_eps);
        }

        let mut relative = self.weights.project(
            &format!("{attn}.wr_du.weight"),
            q_in,
            heads.query * self.math.d_rel,
        )?;
        self.scale_channels(&mut relative, &format!("{attn}.rel_l2_clip.scale"))?;
        clip_rows(
            &mut relative,
            self.math.d_rel,
            self.math.rel_l2_clip_threshold,
        );
        let table = self.weights.vec(
            &format!("{attn}.rel_logits_proj.proj"),
            self.math.d_rel * self.math.rel_extent,
        )?;
        let sinks =
            self.weights
                .project(&format!("{attn}.attn_sink_proj.weight"), q_in, heads.query)?;
        state.push(
            k,
            v,
            match attention {
                Attention::Local => self.spec.sliding_window_size as usize,
                Attention::Global => self.context,
            },
        );

        let mut out = self.attend(state, heads, &q, &relative, &table, &sinks);
        let gates = self.weights.project(
            &format!("{attn}.gate_attn.weight"),
            q_in,
            heads.q_channels(),
        )?;
        for (out, gate) in out.iter_mut().zip(gates) {
            *out *= sigmoid(gate);
        }
        let raw = self
            .weights
            .project(&format!("{attn}.wo_ud.weight"), &out, self.hidden())?;
        self.output_conv(
            raw,
            &format!("{layer}.attn_sconv.weight"),
            &mut state.attn_conv,
        )
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
        for (x, weight) in x.iter_mut().zip(weights) {
            *x *= weight;
        }
        Ok(())
    }

    fn attention_plane(
        &self,
        attn: &str,
        plane: &str,
        input: &[f32],
        heads: Heads,
        attention: Attention,
        conv: &mut CausalConv,
    ) -> Result<Vec<f32>, EngError> {
        let taps = self.math.sconv_kernel_size;
        let channels = heads.kv_channels();
        let raw = self
            .weights
            .project(&format!("{attn}.w{plane}_dv.weight"), input, channels)?;
        let (kernel, dynamic) = match attention {
            Attention::Local => (
                self.weights
                    .vec(&format!("{attn}.{plane}_sconv.weight"), taps * channels)?,
                None,
            ),
            Attention::Global => {
                let kernel = self
                    .weights
                    .vec(&format!("{attn}.{plane}_sconv.static_w"), taps * channels)?;
                let coefficients = self.weights.project(
                    &format!("{attn}.{plane}_sconv.weight_proj"),
                    input,
                    taps * heads.kv,
                )?;
                (
                    kernel,
                    Some(DynamicTaps {
                        coefficients,
                        head_width: heads.width,
                    }),
                )
            }
        };
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
            &mut state.mlp_conv,
        )
    }
}

impl VisitState {
    fn push(&mut self, key: Vec<f32>, value: Vec<f32>, length: usize) {
        self.keys.push_front(Arc::new(key));
        self.values.push_front(Arc::new(value));
        self.keys.truncate(length);
        self.values.truncate(length);
    }
}

fn clip_rows(rows: &mut [f32], width: usize, limit: f32) {
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
