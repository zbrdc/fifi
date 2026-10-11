//! Mac MLX adapter for the Flash-Next projection seam, and oracle parity tests (decision 0035).

use super::stored::{self, StoredWeight};
use crate::family::project::Projector;
use crate::{
    EngError,
    kernels::{MlxKernels, PackedWeight},
    tensor::{Dtype, Tensor},
    weights::StreamedCheckpoint,
};

/// Executes Flash-Next projections with MLX host operations over streamed checkpoint weights.
pub(crate) struct MlxProjector<'c> {
    kernels: MlxKernels,
    ckpt: &'c StreamedCheckpoint,
}

impl<'c> MlxProjector<'c> {
    /// Creates an adapter with its own MLX kernels.
    pub(crate) fn new(ckpt: &'c StreamedCheckpoint) -> Result<Self, EngError> {
        Ok(Self {
            kernels: MlxKernels::try_new()?,
            ckpt,
        })
    }
}

fn backend(message: &str) -> EngError {
    EngError::Backend {
        operation: "flash_next",
        message: message.into(),
    }
}

/// Packs one activation vector as a `[1, len]` f32 host tensor.
fn activation(x: &[f32]) -> Result<Tensor, EngError> {
    let len = u32::try_from(x.len())
        .map_err(|_| EngError::BadRequest("activation has too many values".into()))?;
    let bytes: Vec<u8> = x.iter().flat_map(|value| value.to_le_bytes()).collect();
    Tensor::try_from_host(Dtype::F32, vec![1, len], bytes)
}

/// Reads an f32 projection output as values in row-major order.
fn values(t: &Tensor) -> Result<Vec<f32>, EngError> {
    if t.dtype() != Dtype::F32 {
        return Err(backend("projection output is not f32"));
    }
    let bytes = t
        .host_bytes()
        .ok_or_else(|| backend("projection output has no host bytes"))?;
    let (chunks, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(backend("projection output has a partial value"));
    }
    Ok(chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

/// Returns one logical dimension of a packed weight, or BadRequest when the axis is absent.
fn logical_dim(weight: &PackedWeight, axis: usize) -> Result<usize, EngError> {
    weight
        .logical_shape()
        .get(axis)
        .and_then(|dimension| usize::try_from(*dimension).ok())
        .ok_or_else(|| EngError::BadRequest("packed weight has no such logical dimension".into()))
}

/// Returns the projection output when it holds exactly `want` values.
fn check_output(out: Vec<f32>, want: usize) -> Result<Vec<f32>, EngError> {
    if out.len() == want {
        Ok(out)
    } else {
        Err(backend(
            "projection output length differs from the weight shape",
        ))
    }
}

impl Projector for MlxProjector<'_> {
    type Weight = StoredWeight;

    fn linear(&mut self, weight: &StoredWeight, x: &[f32]) -> Result<Vec<f32>, EngError> {
        match weight {
            StoredWeight::Packed(p) => {
                let out = values(&self.kernels.dequant_matmul(&activation(x)?, p)?)?;
                check_output(out, logical_dim(p, 0)?)
            }
            StoredWeight::Sliced(_) => {
                Err(EngError::BadRequest("linear needs a packed matrix".into()))
            }
        }
    }

    fn expert_linear(
        &mut self,
        stack: &StoredWeight,
        expert: u32,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        match stack {
            StoredWeight::Sliced(s) => {
                let p = stored::read_slice(self.ckpt, s, expert)?;
                let out = values(&self.kernels.dequant_matmul(&activation(x)?, &p)?)?;
                check_output(out, logical_dim(&p, 0)?)
            }
            StoredWeight::Packed(_) => Err(EngError::BadRequest(
                "expert_linear needs a sliced stack".into(),
            )),
        }
    }

    fn rows(&mut self, table: &StoredWeight, rows: &[u32]) -> Result<Vec<f32>, EngError> {
        let s = match table {
            StoredWeight::Sliced(s) => s,
            StoredWeight::Packed(_) => {
                return Err(EngError::BadRequest("rows needs a sliced table".into()));
            }
        };
        let mut out = Vec::new();
        for &row in rows {
            let p = stored::read_slice(self.ckpt, s, row)?;
            let index = Tensor::try_from_host(Dtype::U32, vec![1], 0u32.to_le_bytes().to_vec())?;
            let row_values = values(&self.kernels.quantized_embedding_rows(&p, &index)?)?;
            out.extend(check_output(row_values, logical_dim(&p, 1)?)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::deltanet::{self, DeltaState};
    use super::super::hyper;
    use super::super::load::FlashWeights;
    use super::super::moe;
    use super::super::ple::{self, PleState};
    use super::super::qsa::{self, QsaCache};
    use super::super::step::{self, AttnWeights, FlashLayerWeights, ModelShape, ModelState};
    use super::super::stored::{self, StoredWeight};
    use super::MlxProjector;
    use crate::EngError;
    use crate::family::project::Projector;
    use crate::kernels::{QuantizationMode, QuantizationSpec};
    use crate::weights::{ShardIo, StreamedCheckpoint};
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    pub(in crate::family::flash_next) const PROMPT: [u32; 5] = [760, 6511, 314, 9338, 369];
    pub(in crate::family::flash_next) const COS_BLOCK: f64 = 0.9999;
    pub(in crate::family::flash_next) const REL_BLOCK: f64 = 1e-3;
    pub(in crate::family::flash_next) const COS_END: f64 = 0.999;
    pub(in crate::family::flash_next) const REL_END: f64 = 0.05;

    /// Accumulates gate lines and failing labels for one parity run.
    #[derive(Default)]
    pub(in crate::family::flash_next) struct Gate {
        pub(in crate::family::flash_next) report: String,
        pub(in crate::family::flash_next) failures: Vec<String>,
        pub(in crate::family::flash_next) checks: usize,
    }

    impl Gate {
        /// Compares one output with its oracle row and records the relative error and cosine.
        pub(in crate::family::flash_next) fn check(
            &mut self,
            label: &str,
            got: &[f32],
            want: &[f32],
            cos_min: f64,
            rel_max: f64,
        ) {
            self.checks += 1;
            let mut err = 0.0f64;
            let mut scale = 0.0f64;
            let mut dot = 0.0f64;
            let mut got_sq = 0.0f64;
            let mut want_sq = 0.0f64;
            for (g, w) in got.iter().zip(want) {
                let (g, w) = (f64::from(*g), f64::from(*w));
                err = err.max((g - w).abs());
                scale = scale.max(w.abs());
                dot += g * w;
                got_sq += g * g;
                want_sq += w * w;
            }
            let rel = if scale == 0.0 { err } else { err / scale };
            let cos = if got_sq == 0.0 && want_sq == 0.0 {
                1.0
            } else {
                dot / (got_sq.sqrt() * want_sq.sqrt())
            };
            self.report
                .push_str(&format!("{label} rel={rel:.3e} cos={cos:.7}\n"));
            if got.len() != want.len()
                || cos.is_nan()
                || cos < cos_min
                || rel.is_nan()
                || rel > rel_max
            {
                self.failures.push(label.to_string());
            }
        }

        /// Returns the block value, or records its error under the label.
        pub(in crate::family::flash_next) fn take<T>(
            &mut self,
            label: &str,
            result: Result<T, EngError>,
        ) -> Option<T> {
            match result {
                Ok(value) => Some(value),
                Err(error) => {
                    self.report.push_str(&format!("{label} error={error:?}\n"));
                    self.failures.push(format!("{label}: {error:?}"));
                    None
                }
            }
        }
    }

    /// Attention state for one decoder layer.
    enum AttnState {
        Delta(DeltaState),
        Qsa(QsaCache),
    }

    pub(in crate::family::flash_next) fn env_path(name: &str) -> PathBuf {
        PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set")))
    }

    /// Returns the `weight_map` object of a safetensors index.
    fn weight_map(index: &[u8]) -> serde_json::Map<String, serde_json::Value> {
        let value: serde_json::Value = serde_json::from_slice(index).unwrap();
        value["weight_map"].as_object().unwrap().clone()
    }

    fn index_language_keys(index: &[u8]) -> BTreeSet<String> {
        weight_map(index)
            .keys()
            .filter(|key| key.starts_with("language_model."))
            .cloned()
            .collect()
    }

    pub(in crate::family::flash_next) fn open_checkpoint(dir: &Path) -> StreamedCheckpoint {
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

    /// Reads `{name}.f32` as five equal rows of little-endian f32 values.
    pub(in crate::family::flash_next) fn oracle(dir: &Path, name: &str) -> Vec<Vec<f32>> {
        let bytes = std::fs::read(dir.join(format!("{name}.f32"))).unwrap();
        assert!(
            !bytes.is_empty() && bytes.len().is_multiple_of(20),
            "{name} has {} bytes, which do not split into 5 f32 rows",
            bytes.len()
        );
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let row = values.len() / 5;
        values.chunks(row).map(<[f32]>::to_vec).collect()
    }

    pub(in crate::family::flash_next) fn top5(values: &[f32]) -> Vec<(usize, f32)> {
        let mut ranked: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(5);
        ranked
    }

    pub(in crate::family::flash_next) fn write_report(path: &Path, gate: &Gate) {
        let text = format!("{}failures={}\n", gate.report, gate.failures.len());
        std::fs::write(path, text).unwrap();
    }

    fn attention(
        proj: &mut MlxProjector<'_>,
        shape: &ModelShape,
        layer: &FlashLayerWeights<'_, StoredWeight>,
        state: &mut AttnState,
        pos: u32,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        match (&layer.attn, state) {
            (AttnWeights::Delta(w), AttnState::Delta(s)) => {
                deltanet::step(proj, &shape.delta, w, s, x)
            }
            (AttnWeights::Qsa(w), AttnState::Qsa(c)) => qsa::step(proj, &shape.qsa, w, c, pos, x),
            _ => Err(EngError::BadRequest(
                "attention state does not match its layer".into(),
            )),
        }
    }

    /// Checks one decoder layer over the five prompt tokens, teacher-forced from the oracle rows.
    fn check_layer(
        proj: &mut MlxProjector<'_>,
        shape: &ModelShape,
        layer: &FlashLayerWeights<'_, StoredWeight>,
        dir: &Path,
        index: usize,
        gate: &mut Gate,
    ) {
        let attn_in = oracle(
            dir,
            &format!("read_in.layers.{index}.attn_hyper_connection"),
        );
        let attn_mixed = oracle(dir, &format!("mixed.layers.{index}.attn_hyper_connection"));
        let attn_gate = oracle(dir, &format!("gate.layers.{index}.attn_hyper_connection"));
        let attn_out = oracle(dir, &format!("attn_out.L{index:02}"));
        let mlp_in = oracle(dir, &format!("read_in.layers.{index}.mlp_hyper_connection"));
        let mlp_mixed = oracle(dir, &format!("mixed.layers.{index}.mlp_hyper_connection"));
        let mlp_gate = oracle(dir, &format!("gate.layers.{index}.mlp_hyper_connection"));
        let moe_out = oracle(dir, &format!("moe_out.L{index:02}"));
        let mut attn_state = match &layer.attn {
            AttnWeights::Delta(_) => AttnState::Delta(DeltaState::new(&shape.delta)),
            AttnWeights::Qsa(_) => AttnState::Qsa(QsaCache::new()),
        };
        for (t, attn_row) in attn_in.iter().enumerate() {
            let label = format!("L{index:02} t{t}");
            let Some(read) = gate.take(
                &format!("{label} attn hyper"),
                hyper::read(proj, &shape.hyper, &layer.attn_hyper, attn_row),
            ) else {
                return;
            };
            gate.check(
                &format!("{label} attn hyper mixed"),
                &read.mixed,
                &attn_mixed[t],
                COS_BLOCK,
                REL_BLOCK,
            );
            gate.check(
                &format!("{label} attn hyper gate"),
                read.gate.as_deref().unwrap_or_default(),
                &attn_gate[t],
                COS_BLOCK,
                REL_BLOCK,
            );
            let Some(out) = gate.take(
                &format!("{label} attn block"),
                attention(
                    proj,
                    shape,
                    layer,
                    &mut attn_state,
                    t as u32,
                    &attn_mixed[t],
                ),
            ) else {
                return;
            };
            gate.check(
                &format!("{label} attn out"),
                &out,
                &attn_out[t],
                COS_BLOCK,
                REL_BLOCK,
            );
            let Some(read) = gate.take(
                &format!("{label} mlp hyper"),
                hyper::read(proj, &shape.hyper, &layer.mlp_hyper, &mlp_in[t]),
            ) else {
                return;
            };
            gate.check(
                &format!("{label} mlp hyper mixed"),
                &read.mixed,
                &mlp_mixed[t],
                COS_BLOCK,
                REL_BLOCK,
            );
            gate.check(
                &format!("{label} mlp hyper gate"),
                read.gate.as_deref().unwrap_or_default(),
                &mlp_gate[t],
                COS_BLOCK,
                REL_BLOCK,
            );
            let Some(out) = gate.take(
                &format!("{label} mlp block"),
                moe::step(proj, &shape.moe, &layer.moe, &mlp_mixed[t]),
            ) else {
                return;
            };
            gate.check(
                &format!("{label} mlp out"),
                &out,
                &moe_out[t],
                COS_BLOCK,
                REL_BLOCK,
            );
        }
    }

    #[test]
    #[ignore = "needs the Flash-Next artifact and oracle dump on the Mac"]
    fn blocks_match_the_oracle_teacher_forced() {
        let artifact = env_path("FLASH_NEXT_ARTIFACT");
        let oracle_dir = env_path("FLASH_NEXT_ORACLE");
        let report_path = env_path("FLASH_NEXT_BLOCKS_REPORT");
        let dir = oracle_dir.as_path();
        let ckpt = open_checkpoint(&artifact);
        let config = std::fs::read(artifact.join("config.json")).unwrap();
        let weights = FlashWeights::load(&ckpt, &config).unwrap();
        let view = weights.view();
        let shape = *weights.shape();
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let mut gate = Gate::default();
        let hidden = shape.hyper.hidden();

        let embed_in = oracle(dir, "read_in.layers.0.attn_hyper_connection");
        for (t, (&token, want)) in PROMPT.iter().zip(&embed_in).enumerate() {
            let label = format!("embed t{t}");
            if let Some(got) = gate.take(&label, proj.rows(view.embed, &[token])) {
                gate.check(&label, &got, &want[..hidden], COS_BLOCK, REL_BLOCK);
            }
        }

        assert_eq!(view.layers.len(), 48);
        for (index, layer) in view.layers.iter().enumerate() {
            check_layer(&mut proj, &shape, layer, dir, index, &mut gate);
        }

        let ple_in = oracle(dir, "layer_out.L00");
        let ple_want = oracle(dir, "ple_out");
        let mut ple_state = PleState::new(&shape.ple);
        for (t, (&token, hyper_row)) in PROMPT.iter().zip(&ple_in).enumerate() {
            let label = format!("ple t{t}");
            let result = ple::step(
                &mut proj,
                &shape.ple,
                &view.ple,
                &mut ple_state,
                token,
                hyper_row,
            );
            if let Some(got) = gate.take(&label, result) {
                gate.check(&label, &got, &ple_want[t], COS_BLOCK, REL_BLOCK);
            }
        }

        let mixer_in = oracle(dir, "read_in.hyper_connection_mixer");
        let mixer_mixed = oracle(dir, "mixed.hyper_connection_mixer");
        let logits = oracle(dir, "logits");
        for (t, hin) in mixer_in.iter().enumerate() {
            let label = format!("mixer t{t}");
            let read = hyper::read(&mut proj, &shape.hyper, &view.mixer, hin);
            if let Some(read) = gate.take(&format!("{label} hyper"), read) {
                gate.check(
                    &format!("{label} mixed"),
                    &read.mixed,
                    &mixer_mixed[t],
                    COS_BLOCK,
                    REL_BLOCK,
                );
            }
            let result = proj.linear(view.lm_head, &mixer_mixed[t]);
            if let Some(got) = gate.take(&format!("{label} logits"), result) {
                gate.check(
                    &format!("{label} logits"),
                    &got,
                    &logits[t],
                    COS_BLOCK,
                    REL_BLOCK,
                );
            }
        }

        assert_eq!(gate.checks, 1460, "teacher-forced check count");
        write_report(&report_path, &gate);
        assert!(gate.failures.is_empty(), "{:?}", gate.failures);
    }

    #[test]
    #[ignore = "needs the Flash-Next artifact and oracle dump on the Mac"]
    fn prompt_matches_the_oracle_end_to_end() {
        let artifact = env_path("FLASH_NEXT_ARTIFACT");
        let oracle_dir = env_path("FLASH_NEXT_ORACLE");
        let report_path = env_path("FLASH_NEXT_PROMPT_REPORT");
        let dir = oracle_dir.as_path();
        let index = std::fs::read(artifact.join("model.safetensors.index.json")).unwrap();
        let ckpt = open_checkpoint(&artifact);
        let config = std::fs::read(artifact.join("config.json")).unwrap();
        let mut gate = Gate::default();

        let started = Instant::now();
        let weights = FlashWeights::load(&ckpt, &config).unwrap();
        gate.report.push_str(&format!(
            "load_seconds={:.3}\n",
            started.elapsed().as_secs_f64()
        ));
        let view = weights.view();
        let shape = *weights.shape();
        let mut proj = MlxProjector::new(&ckpt).unwrap();

        let expected = index_language_keys(&index);
        let listed = weights.keys();
        let actual: BTreeSet<String> = listed.iter().cloned().collect();
        let missing: Vec<&String> = expected.difference(&actual).collect();
        let extra: Vec<&String> = actual.difference(&expected).collect();
        gate.report.push_str(&format!(
            "keys expected={} actual={} missing={} extra={}\n",
            expected.len(),
            actual.len(),
            missing.len(),
            extra.len()
        ));
        gate.report.push_str(&format!(
            "keys listed={} unique={}\n",
            listed.len(),
            actual.len()
        ));
        if listed.len() != actual.len() {
            gate.failures.push("duplicate keys".to_string());
        }
        for key in missing.iter().take(20) {
            gate.report.push_str(&format!("missing {key}\n"));
        }
        for key in extra.iter().take(20) {
            gate.report.push_str(&format!("extra {key}\n"));
        }
        if !missing.is_empty() || !extra.is_empty() {
            gate.failures.push("keys".to_string());
        }

        assert_eq!(view.layers.len(), 48);
        let mut state = ModelState::new(&shape, &view);
        let mut outputs = Vec::new();
        for (t, &token) in PROMPT.iter().enumerate() {
            let started = Instant::now();
            let result = step::step(&mut proj, &shape, &view, &mut state, token, true);
            gate.report.push_str(&format!(
                "token{t}_seconds={:.3}\n",
                started.elapsed().as_secs_f64()
            ));
            match gate.take(&format!("token{t}"), result) {
                Some(output) => outputs.push(output),
                None => {
                    write_report(&report_path, &gate);
                    panic!("token {t} failed: {:?}", gate.failures);
                }
            }
        }

        for layer in 0..48usize {
            let want = oracle(dir, &format!("layer_out.L{layer:02}")).concat();
            let got: Vec<f32> = outputs
                .iter()
                .flat_map(|output| {
                    output
                        .layer_out
                        .as_ref()
                        .and_then(|rows| rows.get(layer))
                        .cloned()
                        .unwrap_or_default()
                })
                .collect();
            gate.check(
                &format!("layer_out L{layer:02}"),
                &got,
                &want,
                COS_END,
                REL_END,
            );
        }

        let oracle_logits = oracle(dir, "logits");
        let rust_logits = &outputs[4].logits;
        let want_logits = &oracle_logits[4];
        gate.check("logits t4", rust_logits, want_logits, COS_END, REL_END);
        let rust_top = top5(rust_logits);
        let oracle_top = top5(want_logits);
        gate.report.push_str(&format!(
            "rust_top5={rust_top:?}\noracle_top5={oracle_top:?}\n"
        ));
        let rust_ids: Vec<usize> = rust_top.iter().map(|(id, _)| *id).collect();
        let oracle_ids: Vec<usize> = oracle_top.iter().map(|(id, _)| *id).collect();
        if rust_ids != oracle_ids {
            gate.failures.push("top5 ids".to_string());
        }
        if rust_ids.first() != Some(&11751) {
            gate.failures.push("argmax".to_string());
        }
        for (id, want) in &oracle_top {
            let got = rust_logits.get(*id).copied().unwrap_or(f32::NAN);
            let diff = (got - *want).abs();
            if diff.is_nan() || diff > 0.1 {
                gate.failures.push(format!("top5 value id {id}"));
            }
        }

        write_report(&report_path, &gate);
        assert!(gate.failures.is_empty(), "{:?}", gate.failures);
    }

    /// Builds an in-memory safetensors checkpoint from named tensors.
    fn checkpoint(tensors: &[(&str, &str, Vec<u32>, Vec<u8>)]) -> StreamedCheckpoint {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut index = serde_json::Map::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = payload.len();
            payload.extend_from_slice(bytes);
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
            );
            index.insert(name.to_string(), serde_json::json!("m.safetensors"));
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&payload);
        let index = serde_json::to_vec(&serde_json::json!({"weight_map": index})).unwrap();
        let len = file.len() as u64;
        let reader: Box<dyn ShardIo + Send> = Box::new(std::io::Cursor::new(file));
        StreamedCheckpoint::open(
            &index,
            vec![("m.safetensors".to_string(), reader, len)],
            1 << 20,
        )
        .unwrap()
    }

    /// Returns an in-memory checkpoint with one stack `e` of two 2x32 affine int4 matrices.
    fn stack_checkpoint() -> StreamedCheckpoint {
        checkpoint(&[
            (
                "e.weight",
                "U32",
                vec![2, 2, 4],
                (0..64u8).collect::<Vec<u8>>(),
            ),
            ("e.scales", "BF16", vec![2, 2, 1], [0x80, 0x3f].repeat(4)),
            ("e.biases", "BF16", vec![2, 2, 1], [0x80, 0x3f].repeat(4)),
        ])
    }

    /// Returns the affine int4 layout used by the stack checkpoint.
    fn affine4() -> QuantizationSpec {
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 4,
            group_size: 32,
        }
    }

    #[test]
    fn linear_rejects_a_sliced_weight() -> Result<(), EngError> {
        let ckpt = stack_checkpoint();
        let stack = StoredWeight::Sliced(stored::sliced(&ckpt, "e", affine4(), 2, 32)?);
        let mut proj = MlxProjector::new(&ckpt)?;
        let result = proj.linear(&stack, &[0.0; 32]);
        assert!(matches!(result, Err(EngError::BadRequest(_))), "{result:?}");
        Ok(())
    }

    #[test]
    fn expert_linear_rejects_a_packed_weight() -> Result<(), EngError> {
        let ckpt = stack_checkpoint();
        let sliced = stored::sliced(&ckpt, "e", affine4(), 2, 32)?;
        let packed = StoredWeight::Packed(Box::new(stored::read_slice(&ckpt, &sliced, 0)?));
        let mut proj = MlxProjector::new(&ckpt)?;
        let result = proj.expert_linear(&packed, 0, &[0.0; 32]);
        assert!(matches!(result, Err(EngError::BadRequest(_))), "{result:?}");
        Ok(())
    }

    #[test]
    fn rows_rejects_a_packed_weight() -> Result<(), EngError> {
        let ckpt = stack_checkpoint();
        let sliced = stored::sliced(&ckpt, "e", affine4(), 2, 32)?;
        let packed = StoredWeight::Packed(Box::new(stored::read_slice(&ckpt, &sliced, 0)?));
        let mut proj = MlxProjector::new(&ckpt)?;
        let result = proj.rows(&packed, &[0]);
        assert!(matches!(result, Err(EngError::BadRequest(_))), "{result:?}");
        Ok(())
    }
}
