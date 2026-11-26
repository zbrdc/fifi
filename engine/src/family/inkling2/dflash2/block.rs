//! FP32 CPU reference. Row-major batches; GGUF's first dimension is contiguous.
use super::*;
#[cfg(test)]
mod identity;
use crate::family::speculate::{
    accept::Accepted,
    driver::{Drafter, Proposal, TargetConditioning},
    window::Window,
};
/// Greedy anchor and the selector's scored shortlist at one draft position.
type Selection = (usize, Vec<(usize, f32)>);

impl Drafter for Dflash2 {
    fn propose(&mut self, _: &[u32], _: usize) -> Result<Proposal, EngError> {
        Ok(Proposal::default())
    }
    fn propose_conditioned(
        &mut self,
        history: &[u32],
        room: usize,
        ctx: Option<TargetConditioning<'_>>,
    ) -> Result<Proposal, EngError> {
        let (Some(ctx), Some((&anchor, committed))) = (ctx, history.split_last()) else {
            return Ok(Proposal::default());
        };
        if room == 0 || committed.is_empty() {
            return Ok(Proposal::default());
        }
        if anchor >= 200058 || committed.len() != ctx.features.len() {
            return Err(super::super::bad("DFlash2 target conditioning mismatch"));
        }
        let start = committed.len().saturating_sub(2047);
        let taps = ctx.features[start..].iter().map(|r| r.to_vec()).collect();
        let mut noise = vec![ctx.output.embedding(200056)?; 8];
        noise[0] = ctx.output.embedding(anchor)?;
        let b = Block(&self.0);
        let x = b.hidden(&taps, &noise, start)?;
        let selected = b.select(&x, anchor as usize, |r| ctx.output.project(r))?;
        Ok(Proposal {
            tokens: selected
                .into_iter()
                .take(room)
                .map(|(id, _)| id as u32)
                .collect(),
            scores: Vec::new(),
        })
    }
    fn observe(&mut self, _: &Window, _: &Accepted) -> Result<(), EngError> {
        Ok(())
    }
}
type Rows = Vec<Vec<f32>>;
fn dot(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let mut sums = [0.; 8];
    let (a, ar) = x.as_chunks::<8>();
    let (b, br) = y.as_chunks::<8>();
    for (a, b) in a.iter().zip(b) {
        for k in 0..8 {
            sums[k] += a[k] * b[k];
        }
    }
    sums.into_iter().sum::<f32>() + ar.iter().zip(br).map(|(a, b)| a * b).sum::<f32>()
}
fn norm(x: &mut [f32], w: &[f32]) {
    let s = (dot(x, x) / x.len() as f32 + 1e-6).sqrt().recip();
    for (a, b) in x.iter_mut().zip(w) {
        *a *= s * b;
    }
}
struct Block<'a>(&'a GgufWeights);
impl Block<'_> {
    fn values(&self, n: &str) -> Result<Vec<f32>, EngError> {
        let (t, b) = self.0.read_tensor(n)?;
        crate::weights::dequantize(t, &b)
    }
    fn code(&self, name: &str, id: usize) -> Result<Vec<f32>, EngError> {
        let t = self
            .0
            .header()
            .tensor(name)
            .ok_or_else(|| super::super::bad("missing selector codebook"))?;
        let ty = t.ggml_type();
        let stride = 256 / ty.block_elems() * ty.block_bytes();
        let mut bytes = vec![0; stride as usize];
        self.0.read_range(name, id as u64 * stride, &mut bytes)?;
        crate::weights::dequantize(ty, &bytes)
    }
    fn linear(&self, x: &Rows, n: &str) -> Result<Rows, EngError> {
        let w = self.values(n)?;
        Ok(x.iter()
            .map(|r| w.chunks_exact(r.len()).map(|row| dot(row, r)).collect())
            .collect())
    }
    fn norms(&self, x: &mut Rows, n: &str) -> Result<(), EngError> {
        let w = self.values(n)?;
        for r in x {
            for h in r.chunks_exact_mut(w.len()) {
                norm(h, &w)
            }
        }
        Ok(())
    }
    fn attention(&self, p: &str, x: &Rows, ctx: &Rows, start: usize) -> Result<Rows, EngError> {
        let both = ctx.iter().chain(x).cloned().collect();
        let mut q = self.linear(x, &format!("{p}.attn_q.weight"))?;
        let mut k = self.linear(&both, &format!("{p}.attn_k.weight"))?;
        let v = self.linear(&both, &format!("{p}.attn_v.weight"))?;
        self.norms(&mut q, &format!("{p}.attn_q_norm.weight"))?;
        self.norms(&mut k, &format!("{p}.attn_k_norm.weight"))?;
        rope(&mut q, start + ctx.len());
        rope(&mut k, start);
        let mut out = vec![vec![0.; 4096]; x.len()];
        for (i, row) in q.iter().enumerate() {
            for (h, a) in row.as_chunks::<128>().0.iter().enumerate() {
                let r = h / 4 * 128..(h / 4 + 1) * 128;
                let mut scores: Vec<_> = k
                    .iter()
                    .enumerate()
                    .map(|(j, b)| {
                        if j.abs_diff(ctx.len() + i) < 2048 {
                            dot(a, &b[r.clone()]) / 128f32.sqrt()
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect();
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for s in &mut scores {
                    *s = (*s - m).exp()
                }
                let z = scores.iter().sum::<f32>();
                for (s, b) in scores.iter().zip(&v) {
                    for (dst, val) in out[i][h * 128..(h + 1) * 128].iter_mut().zip(&b[r.clone()]) {
                        *dst += s / z * val;
                    }
                }
            }
        }
        self.linear(&out, &format!("{p}.attn_output.weight"))
    }
    fn hidden(&self, taps: &Rows, noise: &Rows, start: usize) -> Result<Rows, EngError> {
        let mut ctx = self.linear(taps, "fc.weight")?;
        self.norms(&mut ctx, "enc.output_norm.weight")?;
        let mut x = noise.clone();
        for i in 0..5 {
            let p = format!("blk.{i}");
            for op in ["attn", "ffn"] {
                let prefix = format!("{p}.{op}");
                let mut y = x.clone();
                self.norms(&mut y, &format!("{prefix}_norm.weight"))?;
                let d = self.linear(&y, &format!("{prefix}_conv_proj.weight"))?;
                let base = self.values(&format!("{prefix}_conv_base"))?;
                let a = conv(&y, &d, &base, 0);
                let b = if op == "attn" {
                    self.attention(&p, &a, &ctx, start)?
                } else {
                    let mut gate = self.linear(&a, &format!("{p}.ffn_gate.weight"))?;
                    let up = self.linear(&a, &format!("{p}.ffn_up.weight"))?;
                    for (g, u) in gate.iter_mut().flatten().zip(up.iter().flatten()) {
                        *g = *g / (1. + (-*g).exp()) * u;
                    }
                    self.linear(&gate, &format!("{p}.ffn_down.weight"))?
                };
                for (r, v) in x
                    .iter_mut()
                    .flatten()
                    .zip(conv(&b, &d, &base, 1).into_iter().flatten())
                {
                    *r += v;
                }
            }
        }
        self.norms(&mut x, "output_norm.weight")?;
        Ok(x)
    }
}
fn rope(rows: &mut Rows, start: usize) {
    for (t, row) in rows.iter_mut().enumerate() {
        for h in row.as_chunks_mut::<128>().0 {
            for j in 0..64 {
                let (s, c) = ((t + start) as f32 * 10000f32.powf(-(j as f32) / 64.)).sin_cos();
                let (a, b) = (h[j], h[64 + j]);
                h[j] = a * c - b * s;
                h[64 + j] = b * c + a * s;
            }
        }
    }
}
fn conv(x: &Rows, d: &Rows, b: &[f32], side: usize) -> Rows {
    let mut y = x.clone();
    for (i, row) in y.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = 0.;
            for tap in 0..=1.min(i) {
                let k = 2 * side + tap;
                *v += (b[k * 2304 + j] + d[i][k * 144 + j / 16]) * x[i - tap][j];
            }
        }
    }
    y
}
impl Block<'_> {
    fn select(
        &self,
        x: &Rows,
        mut anchor: usize,
        project: impl Fn(&[f32]) -> Result<Vec<f32>, EngError>,
    ) -> Result<Vec<Selection>, EngError> {
        let proj = self.linear(x, "selector_hidden.weight")?;
        let mut output = Vec::new();
        for i in 1..x.len() {
            let mut scores: Vec<_> = project(&x[i])?.into_iter().enumerate().collect();
            scores.sort_by(|a, b| b.1.total_cmp(&a.1));
            scores.truncate(16);
            let mut best = (0, f32::NEG_INFINITY);
            let prev = self.code("selector_predecessor.weight", anchor)?;
            for (id, s) in &mut scores {
                let next = self.code("selector_successor.weight", *id)?;
                let c: f32 = (0..256).map(|k| proj[i][k] * prev[k] * next[k]).sum();
                *s = 30. * ((*s + c) / 30.).tanh();
                if *s > best.1 {
                    best = (*id, *s);
                }
            }
            anchor = best.0;
            output.push((anchor, scores));
        }
        Ok(output)
    }
}
#[test]
#[ignore = "requires real BF16 conversion"]
fn block_oracle() {
    let path = std::env::var_os("FIFI_INKLING_SPEC_DRAFT").unwrap();
    let w = super::open(Path::new(&path)).unwrap();
    let model = Block(&w);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/dflash2");
    let rows = |n: &str, width: usize| -> Rows {
        let b = std::fs::read(root.join(n)).unwrap();
        b.chunks_exact(width * 4)
            .map(|r| {
                r.as_chunks::<4>()
                    .0
                    .iter()
                    .copied()
                    .map(f32::from_le_bytes)
                    .collect()
            })
            .collect()
    };
    let x = model
        .hidden(&rows("taps.f32le", 9216), &rows("noise.f32le", 2304), 0)
        .unwrap();
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("expected.json")).unwrap()).unwrap();
    let head = rows("head.f32le", 2304);
    let selected = model
        .select(&x, 33, |r| Ok(head.iter().map(|w| dot(w, r)).collect()))
        .unwrap();
    let mut delta = 0f32;
    for (i, (anchor, scores)) in selected.into_iter().enumerate() {
        for (j, (id, s)) in scores.into_iter().enumerate() {
            assert_eq!(id, expected["candidates"][i][j].as_u64().unwrap() as usize);
            delta = delta.max((s - expected["scores"][i][j].as_f64().unwrap() as f32).abs());
        }
        assert_eq!(anchor, expected["tokens"][i].as_u64().unwrap() as usize);
    }
    println!("max selector error {delta}");
    assert!(delta < 1e-3);
}
