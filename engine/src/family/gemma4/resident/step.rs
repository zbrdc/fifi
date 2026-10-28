//! Decode and bounded prompt batches of the resident Gemma 4 Humanizer model.

use super::{CELLS, FFN, Gemma, HEADS, HIDDEN, LOCAL_WINDOW, SOFT_CAP, VOCAB};
use crate::EngError;
use crate::kernels::cuda::DeviceRange;
use crate::kernels::cuda::resident::add_rmsnorm_q8::AddNormQ8Call;
use crate::kernels::cuda::resident::attn_decode_gemma::AttnDecodeGemmaCall;
use crate::kernels::cuda::resident::embed_q8::EmbedQ8Call;
use crate::kernels::cuda::resident::gelu_ffn::GeluFfnCall;
use crate::kernels::cuda::resident::gemma_matvec::GemmaMatvecCall;
use crate::kernels::cuda::resident::kv8_append::Kv8Data;
use crate::kernels::cuda::resident::matvec_q::{MatvecCall, QuantType};
use crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall;
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
use crate::kernels::cuda::resident::rope_gemma::RopeGemmaCall;
use crate::kernels::cuda::resident::row_add::RowAddCall;
use crate::kernels::cuda::resident::row_scale::RowScaleCall;
use crate::kernels::cuda::resident::softcap::SoftcapCall;

/// Scale applied to the embedding row: sqrt(hidden).
pub(crate) fn embed_scale() -> f32 {
    (HIDDEN as f32).sqrt()
}

/// Which residual buffers one layer reads and writes.
/// `current` is the buffer holding the finished output of the previous layer; the layer
/// writes `out`, and the next layer reads `out`, so the rotation is data, not arithmetic.
pub(super) fn layer_buffers(current: usize) -> (usize, usize, usize, usize) {
    match current {
        1 => (1, 2, 3, 0),
        2 => (2, 3, 0, 1),
        3 => (3, 0, 1, 2),
        _ => (0, 1, 2, 3),
    }
}

impl Gemma {
    pub(crate) const PREFILL_BATCH: usize = 8;

    /// Runs a bounded prompt batch. Only its last row can produce logits.
    /// Scratch grows once; each batch drains the stream exactly once after all layers.
    pub(crate) fn prefill_batch(
        &mut self,
        tokens: &[u32],
        pos: usize,
        logits: bool,
    ) -> Result<(), EngError> {
        if !(1..=Self::PREFILL_BATCH).contains(&tokens.len()) {
            return Err(EngError::BadRequest(
                "a Gemma prefill batch needs 1 to 8 tokens".to_owned(),
            ));
        }
        if pos.checked_add(tokens.len()).is_none_or(|end| end > CELLS) {
            return Err(EngError::ContextLimit(
                "the Gemma prefill batch exceeds the context".to_owned(),
            ));
        }
        self.grow_scratch(tokens.len())?;
        self.decode_with_logits(tokens, pos, logits)?;
        if !logits {
            self.stream.synchronize()?;
        }
        Ok(())
    }

    fn grow_scratch(&mut self, rows: usize) -> Result<(), EngError> {
        let sc = &mut self.scratch;
        for (v, width) in sc.h.iter_mut().map(|v| (v, HIDDEN)).chain([
            (&mut sc.nrm, HIDDEN),
            (&mut sc.nrm2, HIDDEN),
            (&mut sc.q_normed, HEADS * 512),
            (&mut sc.k_normed, 8 * 256),
            (&mut sc.v_normed, 8 * 256),
            (&mut sc.q, HEADS * 512),
            (&mut sc.k, 8 * 256),
            (&mut sc.v, 8 * 256),
            (&mut sc.ctx_out, HEADS * 512),
            (&mut sc.proj, HIDDEN),
            (&mut sc.gate, FFN),
            (&mut sc.up, FFN),
            (&mut sc.mixed, FFN),
            (&mut sc.down, HIDDEN),
        ]) {
            if v.len < rows * width {
                let len = Self::PREFILL_BATCH * width;
                v.buf = self.ctx.allocate(len * 4)?;
                v.len = len;
            }
        }
        for (buffer, width) in [
            (&mut sc.q8_hidden, HIDDEN),
            (&mut sc.q8_ctx, HEADS * 512),
            (&mut sc.q8_mixed, FFN),
        ] {
            if buffer.bytes() < q8_1_bytes(width, rows).expect("bounded q8 rows") {
                *buffer = self
                    .ctx
                    .allocate(q8_1_bytes(width, Self::PREFILL_BATCH).expect("bounded q8 rows"))?;
            }
        }
        Ok(())
    }

    /// Keeps the existing norm reduction and quantization order for every activation row.
    fn norm_rows<'b>(&self, call: AddNormQ8Call<'b>, rows: usize) -> Result<(), EngError> {
        for row in 0..rows {
            let slice = |range: DeviceRange<'b>, bytes| {
                range.buffer().range(range.offset() + row * bytes, bytes)
            };
            self.ctx.add_rmsnorm_q8(
                &self.stream,
                AddNormQ8Call {
                    n: call.n,
                    eps: call.eps,
                    h: slice(call.h, call.n * 4)?,
                    delta: call.delta.map(|r| slice(r, call.n * 4)).transpose()?,
                    weight: call.weight,
                    norm_out: slice(call.norm_out, call.n * 4)?,
                    q8: slice(call.q8, q8_1_bytes(call.n, 1).expect("bounded q8 row"))?,
                },
            )?;
        }
        Ok(())
    }

    /// Runs one decode step for the token at the position and leaves the capped
    /// logits in the device buffer. Synchronizes once at the end.
    pub(crate) fn decode(&mut self, token: u32, pos: usize) -> Result<(), EngError> {
        self.decode_with_logits(&[token], pos, true)
    }

    /// Runs one step without the output norm and the logits projection, for prompt tokens
    /// whose logits are never read. It queues the step and does not synchronize.
    pub(crate) fn decode_hidden(&mut self, token: u32, pos: usize) -> Result<(), EngError> {
        self.decode_with_logits(&[token], pos, false)
    }

    fn decode_with_logits(
        &mut self,
        tokens: &[u32],
        pos: usize,
        logits: bool,
    ) -> Result<(), EngError> {
        let m = tokens.len();
        let ids: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        self.scratch.ids = self.ctx.upload(&ids)?;
        self.scratch.pos = self.ctx.upload(&(pos as i32).to_le_bytes())?;
        self.scratch.kv_len = self.ctx.upload(&((pos as i32) + 1).to_le_bytes())?;
        let ctx = &self.ctx;
        let stream = &self.stream;
        let sc = &self.scratch;
        let h0 = &sc.h[0];
        let h1 = &sc.h[1];
        let h2 = &sc.h[2];
        let h3 = &sc.h[3];
        ctx.embed_q8(
            stream,
            EmbedQ8Call {
                out: h0.buf.range(0, h0.bytes())?,
                w: self.embed.buf.range(0, self.embed.bytes())?,
                ids: sc.ids.range(0, m * 4)?,
                m,
                n_rows: self.embed.rows,
                k: self.embed.k,
            },
        )?;
        ctx.row_scale(
            stream,
            RowScaleCall {
                out: h1.buf.range(0, h1.bytes())?,
                x: h0.buf.range(0, h0.bytes())?,
                scale: embed_scale(),
                m,
                n: HIDDEN,
            },
        )?;
        let h = [h0, h1, h2, h3];
        let mut current = 1usize;
        for layer in &self.layers {
            let (input, r1, r2, out) = layer_buffers(current);
            self.decode_layer(layer, h[input], h[r1], h[r2], h[out], pos, m)?;
            // decode_layer leaves the finished layer output in the out buffer, so the
            // next layer reads exactly that buffer.
            current = out;
        }
        if !logits {
            // Only the last prompt token contributes logits, so the output norm and the
            // 1.07 GB tied-embedding projection are skipped for the rest.
            return Ok(());
        }
        let input = h[current];
        ctx.add_rmsnorm_q8(
            stream,
            AddNormQ8Call {
                n: HIDDEN,
                eps: self.eps,
                h: input.buf.range((m - 1) * HIDDEN * 4, HIDDEN * 4)?,
                delta: None,
                weight: self.out_norm.buf.range(0, self.out_norm.bytes())?,
                norm_out: sc.nrm.buf.range(0, sc.nrm.bytes())?,
                q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            },
        )?;
        ctx.matvec_q(
            stream,
            MatvecCall {
                ty: QuantType::Q80,
                w: self.embed.buf.range(0, self.embed.bytes())?,
                x_q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                y: self.logits.buf.range(0, self.logits.bytes())?,
                k: HIDDEN,
                rows: self.embed.rows,
                ncols: 1,
            },
        )?;
        ctx.softcap(
            stream,
            SoftcapCall {
                out: self
                    .logits_capped
                    .buf
                    .range(0, self.logits_capped.bytes())?,
                x: self.logits.buf.range(0, self.logits.bytes())?,
                cap: SOFT_CAP,
                n: VOCAB,
            },
        )?;
        self.stream.synchronize()
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the residual rotation keeps four separate buffers"
    )]
    fn decode_layer(
        &self,
        layer: &super::Layer,
        input: &super::Vector,
        r1: &super::Vector,
        r2: &super::Vector,
        out: &super::Vector,
        pos: usize,
        m: usize,
    ) -> Result<(), EngError> {
        let ctx = &self.ctx;
        let stream = &self.stream;
        let sc = &self.scratch;
        let head = layer.head;
        let n_kv = layer.n_kv;
        let q_rows = HEADS * head;
        let kv_rows = n_kv * head;
        self.norm_rows(
            AddNormQ8Call {
                n: HIDDEN,
                eps: self.eps,
                h: input.buf.range(0, input.bytes())?,
                delta: None,
                weight: layer.attn_norm.buf.range(0, layer.attn_norm.bytes())?,
                norm_out: sc.nrm.buf.range(0, sc.nrm.bytes())?,
                q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            },
            m,
        )?;
        ctx.gemma_matvec(
            stream,
            GemmaMatvecCall {
                w_qs: layer.q.qs.range(0, layer.q.qs_bytes())?,
                w_scales: layer.q.scales.range(0, layer.q.scales_bytes())?,
                x_q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                y: sc.q.buf.range(0, m * q_rows * 4)?,
                k: HIDDEN,
                rows: q_rows,
                ncols: m,
            },
        )?;
        ctx.gemma_matvec(
            stream,
            GemmaMatvecCall {
                w_qs: layer.k.qs.range(0, layer.k.qs_bytes())?,
                w_scales: layer.k.scales.range(0, layer.k.scales_bytes())?,
                x_q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                y: sc.k.buf.range(0, m * kv_rows * 4)?,
                k: HIDDEN,
                rows: kv_rows,
                ncols: m,
            },
        )?;
        match &layer.v {
            Some(v) => ctx.gemma_matvec(
                stream,
                GemmaMatvecCall {
                    w_qs: v.qs.range(0, v.qs_bytes())?,
                    w_scales: v.scales.range(0, v.scales_bytes())?,
                    x_q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                    y: sc.v.buf.range(0, m * kv_rows * 4)?,
                    k: HIDDEN,
                    rows: kv_rows,
                    ncols: m,
                },
            )?,
            None => ctx.gemma_matvec(
                stream,
                GemmaMatvecCall {
                    w_qs: layer.k.qs.range(0, layer.k.qs_bytes())?,
                    w_scales: layer.k.scales.range(0, layer.k.scales_bytes())?,
                    x_q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                    y: sc.v.buf.range(0, m * kv_rows * 4)?,
                    k: HIDDEN,
                    rows: kv_rows,
                    ncols: m,
                },
            )?,
        }
        // The QK-norm of the reference graph: each head of q and of k is
        // RMS-normalized and scaled by its own weight before RoPE. The raw
        // matvec output stays in sc.q/sc.k and the normalized rows land in the
        // q_normed/k_normed buffers, which RoPE then rotates in place. Values get the
        // reference's unweighted per-head RMS norm below, before their cache write.
        for h in 0..m * HEADS {
            ctx.add_rmsnorm_q8(
                stream,
                AddNormQ8Call {
                    n: head,
                    eps: self.eps,
                    h: sc.q.buf.range(h * head * 4, head * 4)?,
                    delta: None,
                    weight: layer.q_norm.buf.range(0, head * 4)?,
                    norm_out: sc.q_normed.buf.range(h * head * 4, head * 4)?,
                    q8: sc.q8_head.range(0, sc.q8_head.bytes())?,
                },
            )?;
        }
        for h in 0..m * n_kv {
            ctx.add_rmsnorm_q8(
                stream,
                AddNormQ8Call {
                    n: head,
                    eps: self.eps,
                    h: sc.k.buf.range(h * head * 4, head * 4)?,
                    delta: None,
                    weight: layer.k_norm.buf.range(0, head * 4)?,
                    norm_out: sc.k_normed.buf.range(h * head * 4, head * 4)?,
                    q8: sc.q8_head.range(0, sc.q8_head.bytes())?,
                },
            )?;
        }
        // The reference normalizes values too: unweighted per-head RMS norm, applied
        // before the cache write and with no rotation. The ones vector makes the
        // weighted kernel compute the unweighted norm.
        for h in 0..m * n_kv {
            ctx.add_rmsnorm_q8(
                stream,
                AddNormQ8Call {
                    n: head,
                    eps: self.eps,
                    h: sc.v.buf.range(h * head * 4, head * 4)?,
                    delta: None,
                    weight: sc.ones.buf.range(0, head * 4)?,
                    norm_out: sc.v_normed.buf.range(h * head * 4, head * 4)?,
                    q8: sc.q8_head.range(0, sc.q8_head.bytes())?,
                },
            )?;
        }
        ctx.rope_gemma(
            stream,
            RopeGemmaCall {
                q: sc.q_normed.buf.range(0, m * q_rows * 4)?,
                k: sc.k_normed.buf.range(0, m * kv_rows * 4)?,
                pos: sc.pos.range(0, 4)?,
                factors: match (&self.rope_freqs, layer.global) {
                    (Some(freqs), true) => Some(freqs.buf.range(0, freqs.bytes())?),
                    _ => None,
                },
                rows_q: m,
                rows_k: m,
                n_q_head: HEADS,
                n_kv_head: n_kv,
                head_dim: head,
                global_mode: layer.global,
            },
        )?;
        ctx.kv8_chunk(
            stream,
            crate::kernels::cuda::resident::kv8_append::Kv8Shape {
                rows: m,
                ..layer.kv.shape
            },
            Kv8Data {
                k: sc.k_normed.buf.range(0, m * kv_rows * 4)?,
                v: sc.v_normed.buf.range(0, m * kv_rows * 4)?,
                k_codes: layer.kv.k_codes.range(0, layer.kv.k_codes.bytes())?,
                k_scales: layer.kv.k_scales.range(0, layer.kv.k_scales.bytes())?,
                v_codes: layer.kv.v_codes.range(0, layer.kv.v_codes.bytes())?,
                v_scales: layer.kv.v_scales.range(0, layer.kv.v_scales.bytes())?,
            },
            pos,
        )?;
        ctx.attn_decode_gemma(
            stream,
            AttnDecodeGemmaCall {
                out: sc.ctx_out.buf.range(0, m * q_rows * 4)?,
                status: None,
                q: sc.q_normed.buf.range(0, m * q_rows * 4)?,
                k_codes: layer.kv.k_codes.range(0, layer.kv.k_codes.bytes())?,
                k_scales: layer.kv.k_scales.range(0, layer.kv.k_scales.bytes())?,
                v_codes: layer.kv.v_codes.range(0, layer.kv.v_codes.bytes())?,
                v_scales: layer.kv.v_scales.range(0, layer.kv.v_scales.bytes())?,
                kv_len: sc.kv_len.range(0, 4)?,
                m,
                n_q_head: HEADS,
                n_kv_head: n_kv,
                head_dim: head,
                cells: CELLS,
                window: if layer.global { 0 } else { LOCAL_WINDOW },
                scale: 1.0,
            },
        )?;
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: sc.ctx_out.buf.range(0, m * q_rows * 4)?,
                y: sc.q8_ctx.range(0, sc.q8_ctx.bytes())?,
                k: q_rows,
                ncols: m,
            },
        )?;
        ctx.gemma_matvec(
            stream,
            GemmaMatvecCall {
                w_qs: layer.out.qs.range(0, layer.out.qs_bytes())?,
                w_scales: layer.out.scales.range(0, layer.out.scales_bytes())?,
                x_q8: sc.q8_ctx.range(0, sc.q8_ctx.bytes())?,
                y: sc.proj.buf.range(0, m * HIDDEN * 4)?,
                k: q_rows,
                rows: HIDDEN,
                ncols: m,
            },
        )?;
        self.norm_rows(
            AddNormQ8Call {
                n: HIDDEN,
                eps: self.eps,
                h: sc.proj.buf.range(0, m * HIDDEN * 4)?,
                delta: None,
                weight: layer
                    .post_attn_norm
                    .buf
                    .range(0, layer.post_attn_norm.bytes())?,
                norm_out: sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
                q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            },
            m,
        )?;
        ctx.row_add(
            stream,
            RowAddCall {
                out: r1.buf.range(0, r1.bytes())?,
                x: input.buf.range(0, input.bytes())?,
                y: sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
                n: m * HIDDEN,
            },
        )?;
        self.norm_rows(
            AddNormQ8Call {
                n: HIDDEN,
                eps: self.eps,
                h: r1.buf.range(0, r1.bytes())?,
                delta: None,
                weight: layer.ffn_norm.buf.range(0, layer.ffn_norm.bytes())?,
                norm_out: sc.nrm.buf.range(0, sc.nrm.bytes())?,
                q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            },
            m,
        )?;
        for (weight, y) in [(&layer.gate, &sc.gate), (&layer.up, &sc.up)] {
            ctx.gemma_matvec(
                stream,
                GemmaMatvecCall {
                    w_qs: weight.qs.range(0, weight.qs_bytes())?,
                    w_scales: weight.scales.range(0, weight.scales_bytes())?,
                    x_q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                    y: y.buf.range(0, m * FFN * 4)?,
                    k: HIDDEN,
                    rows: FFN,
                    ncols: m,
                },
            )?;
        }
        ctx.gelu_ffn(
            stream,
            GeluFfnCall {
                out: sc.mixed.buf.range(0, m * FFN * 4)?,
                gate: sc.gate.buf.range(0, m * FFN * 4)?,
                up: sc.up.buf.range(0, m * FFN * 4)?,
                m,
                n: FFN,
            },
        )?;
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: sc.mixed.buf.range(0, m * FFN * 4)?,
                y: sc.q8_mixed.range(0, sc.q8_mixed.bytes())?,
                k: FFN,
                ncols: m,
            },
        )?;
        ctx.gemma_matvec(
            stream,
            GemmaMatvecCall {
                w_qs: layer.down.qs.range(0, layer.down.qs_bytes())?,
                w_scales: layer.down.scales.range(0, layer.down.scales_bytes())?,
                x_q8: sc.q8_mixed.range(0, sc.q8_mixed.bytes())?,
                y: sc.down.buf.range(0, m * HIDDEN * 4)?,
                k: FFN,
                rows: HIDDEN,
                ncols: m,
            },
        )?;
        self.norm_rows(
            AddNormQ8Call {
                n: HIDDEN,
                eps: self.eps,
                h: sc.down.buf.range(0, m * HIDDEN * 4)?,
                delta: None,
                weight: layer
                    .post_ffw_norm
                    .buf
                    .range(0, layer.post_ffw_norm.bytes())?,
                norm_out: sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
                q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            },
            m,
        )?;
        ctx.row_add(
            stream,
            RowAddCall {
                out: r2.buf.range(0, r2.bytes())?,
                x: r1.buf.range(0, r1.bytes())?,
                y: sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
                n: m * HIDDEN,
            },
        )?;
        ctx.row_scale(
            stream,
            RowScaleCall {
                out: out.buf.range(0, out.bytes())?,
                x: r2.buf.range(0, r2.bytes())?,
                scale: layer.scale,
                m,
                n: HIDDEN,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::resident::sample_gemma::{SampleGemmaCall, SampleParams};

    fn logits(model: &Gemma) -> Vec<f32> {
        let mut bytes = vec![0u8; VOCAB * 4];
        model
            .ctx
            .download(&model.logits_capped.buf, &mut bytes)
            .expect("logits");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    fn sample(model: &Gemma, position: usize) -> u32 {
        let ctx = &model.ctx;
        let token = ctx.allocate(4).expect("token");
        let logprob = ctx.allocate(4).expect("log probability");
        let status = ctx.upload(&[0u8; 4]).expect("status");
        let probs = ctx.allocate(VOCAB * 4).expect("probabilities");
        ctx.sample_gemma(
            &model.stream,
            SampleGemmaCall {
                token: token.range(0, 4).expect("token"),
                logprob: logprob.range(0, 4).expect("log probability"),
                status: status.range(0, 4).expect("status"),
                probs: probs.range(0, probs.bytes()).expect("probabilities"),
                logits: model.logits_capped.buf.range(0, VOCAB * 4).expect("logits"),
                n: VOCAB,
                params: SampleParams {
                    temperature: 1.0,
                    top_k: 0,
                    top_p: 0.95,
                    min_p: 0.0,
                    seed: 0,
                    position: position as u64,
                },
            },
        )
        .expect("sample");
        model.stream.synchronize().expect("sample completion");
        let mut bytes = [0u8; 4];
        ctx.download(&status, &mut bytes).expect("sample status");
        assert_eq!(u32::from_le_bytes(bytes), 0, "sampler status");
        ctx.download(&token, &mut bytes).expect("sampled token");
        u32::from_le_bytes(bytes)
    }

    fn assert_logits(got: &[f32], want: &[f32]) {
        let mut max_relative = 0.0f32;
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(g.is_finite() && w.is_finite(), "non-finite logit {i}");
            let error = (g - w).abs();
            assert!(error <= 2e-5 * w.abs(), "logit {i}: {g} versus {w}");
            if w != 0.0 {
                max_relative = max_relative.max(error / w.abs());
            }
        }
        eprintln!("logits max relative error {max_relative}");
    }

    /// One model owns the device. Replaying from position zero overwrites every live cache cell.
    #[test]
    #[ignore = "requires the pinned Humanizer file and Xavier; use --release --test-threads=1"]
    fn batched_prefill_matches_single_steps() {
        let path = std::env::var("GEMMA4_Q8_FILE").expect("pinned model path");
        let header = std::env::var("GEMMA4_Q8_HEADER_SHA256").expect("pinned header hash");
        let mut model =
            super::super::load(std::path::Path::new(&path), Some(&header)).expect("load");
        assert!(matches!(
            model.prefill_batch(&[], 0, false),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            model.prefill_batch(&[2; 9], 0, false),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            model.prefill_batch(&[2], CELLS, false),
            Err(EngError::ContextLimit(_))
        ));
        assert!(matches!(
            model.prefill_batch(&[2], usize::MAX, false),
            Err(EngError::ContextLimit(_))
        ));
        let prompt: Vec<u32> = (0..32).map(|i| 2 + i * 17).collect();
        let single_started = std::time::Instant::now();
        for (pos, &token) in prompt.iter().enumerate() {
            if pos + 1 == prompt.len() {
                model.decode(token, pos).expect("single last token");
            } else {
                model
                    .decode_hidden(token, pos)
                    .expect("single hidden token");
            }
        }
        let single_elapsed = single_started.elapsed();
        let expected = logits(&model);
        let expected_token = sample(&model, prompt.len());
        model
            .decode(expected_token, prompt.len())
            .expect("single continuation");
        let continued = logits(&model);
        eprintln!(
            "timing 32 single steps {single_elapsed:?}; first sampled token {expected_token}"
        );
        // Eight rows is the shipped cap; three also exercises a partial final batch.
        for width in [1, Gemma::PREFILL_BATCH, 3] {
            let started = std::time::Instant::now();
            let mut pos = 0;
            for batch in prompt.chunks(width) {
                model
                    .prefill_batch(batch, pos, pos + batch.len() == prompt.len())
                    .expect("batch");
                pos += batch.len();
            }
            let elapsed = started.elapsed();
            assert_logits(&logits(&model), &expected);
            assert_eq!(
                sample(&model, prompt.len()),
                expected_token,
                "width {width}"
            );
            model
                .decode(expected_token, prompt.len())
                .expect("batch continuation");
            assert_logits(&logits(&model), &continued);
            eprintln!(
                "timing 32 batched tokens m={width} {elapsed:?}; speedup {:.3}",
                single_elapsed.as_secs_f64() / elapsed.as_secs_f64()
            );
        }
    }
}
