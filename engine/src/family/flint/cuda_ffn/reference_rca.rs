//! Test-only L0 boundary diagnostics. None of these schemes changes qualification.

use super::*;
use crate::family::flint::{
    cuda_ffn::{real_model, reference::bf16},
    reference::AttentionPartial,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Scheme {
    FrozenF16,
    TwoRmsF16,
    RecBoundariesBf16,
    Fp64,
}

impl Scheme {
    fn publish(self, value: f64) -> Result<f64, String> {
        if !value.is_finite() {
            return Err("nonfinite RCA intermediate".into());
        }
        if matches!(self, Self::Fp64) {
            Ok(value)
        } else {
            bf16(value as f32).map(f64::from)
        }
    }
}

#[derive(Serialize)]
struct Boundary {
    name: &'static str,
    columns: usize,
    raw: Vec<f64>,
    values: Vec<f64>,
}

#[derive(Serialize)]
struct Trace {
    scheme: Scheme,
    rows: usize,
    boundaries: Vec<Boundary>,
}

impl Trace {
    fn record(
        &mut self,
        name: &'static str,
        columns: usize,
        raw: Vec<f64>,
    ) -> Result<Vec<f64>, String> {
        if columns == 0 || self.rows.checked_mul(columns) != Some(raw.len()) || raw.is_empty() {
            return Err("invalid RCA boundary geometry".into());
        }
        let values = raw
            .iter()
            .map(|&v| self.scheme.publish(v))
            .collect::<Result<Vec<_>, _>>()?;
        self.boundaries.push(Boundary {
            name,
            columns,
            raw,
            values: values.clone(),
        });
        Ok(values)
    }

    fn norm(
        &mut self,
        name: &'static str,
        input: &[f64],
        gamma: &[f32],
    ) -> Result<Vec<f64>, String> {
        if gamma.is_empty() || input.is_empty() || !input.len().is_multiple_of(gamma.len()) {
            return Err("invalid RCA normalization geometry".into());
        }
        let mut raw = Vec::with_capacity(input.len());
        for row in input.chunks_exact(gamma.len()) {
            let inverse = (row.iter().map(|v| v * v).sum::<f64>() / gamma.len() as f64
                + f64::from(1e-6_f32))
            .sqrt()
            .recip();
            for (&value, &weight) in row.iter().zip(gamma) {
                let unit = value * inverse;
                let unit = if matches!(self.scheme, Scheme::TwoRmsF16 | Scheme::RecBoundariesBf16) {
                    self.scheme.publish(unit)?
                } else {
                    unit
                };
                let weight = if matches!(self.scheme, Scheme::RecBoundariesBf16) {
                    f64::from(bf16(weight)?)
                } else {
                    f64::from(weight)
                };
                raw.push(unit * weight);
            }
        }
        self.record(name, input.len() / self.rows, raw)
    }

    fn project(
        &mut self,
        name: &'static str,
        input: &[f64],
        matrix: &Matrix,
    ) -> Result<Vec<f64>, String> {
        validate_matrix(matrix)?;
        if input.is_empty() || !input.len().is_multiple_of(matrix.cols) {
            return Err("invalid RCA projection geometry".into());
        }
        let mut raw = Vec::with_capacity(input.len() / matrix.cols * matrix.rows);
        for row in input.chunks_exact(matrix.cols) {
            for weight in matrix.values.chunks_exact(matrix.cols) {
                raw.push(
                    row.iter()
                        .zip(weight)
                        .map(|(&x, &w)| x * f64::from(w))
                        .sum::<f64>()
                        * f64::from(matrix.factor),
                );
            }
        }
        self.record(name, matrix.rows, raw)
    }

    fn cache(&mut self, name: &'static str, input: &[f64]) -> Result<Vec<f64>, String> {
        if self.rows.checked_mul(512) != Some(input.len()) || input.is_empty() {
            return Err("invalid RCA physical cache geometry".into());
        }
        let values = if matches!(self.scheme, Scheme::FrozenF16 | Scheme::TwoRmsF16) {
            input
                .iter()
                .map(|&v| f64::from(half::f16::from_f32(v as f32).to_f32()))
                .collect::<Vec<_>>()
        } else {
            input.to_vec()
        };
        if values.iter().any(|v| !v.is_finite()) {
            return Err("RCA physical cache overflow".into());
        }
        self.boundaries.push(Boundary {
            name,
            columns: 512,
            raw: input.to_vec(),
            values: values.clone(),
        });
        Ok(values)
    }

    fn attention(&mut self, qkv: [&[f64]; 3], sinks: &[f32]) -> Result<Vec<f64>, String> {
        let [q, k, v] = qkv;
        let rows = q.len() / 4096;
        if rows == 0
            || q.len() != rows * 4096
            || k.len() != rows * 512
            || v.len() != k.len()
            || sinks.len() != 32
        {
            return Err("invalid RCA attention geometry".into());
        }
        // Preserve the frozen F32 coefficient; only intermediate arithmetic/publications differ.
        let scale = f64::from(128.0_f32.sqrt().recip());
        let mut raw = Vec::with_capacity(q.len());
        for token in 0..rows {
            for (head, &sink) in sinks.iter().enumerate() {
                let query = &q[(token * 32 + head) * 128..][..128];
                let mut partial = AttentionPartial::new(128).map_err(|e| format!("{e:?}"))?;
                for key in 0..=token {
                    let start = (key * 4 + head / 8) * 128;
                    let score = query
                        .iter()
                        .zip(&k[start..start + 128])
                        .map(|(&q, &k)| q * k)
                        .sum::<f64>()
                        * scale;
                    partial
                        .push(score, &v[start..start + 128])
                        .map_err(|e| format!("{e:?}"))?;
                }
                let sink = if matches!(self.scheme, Scheme::RecBoundariesBf16) {
                    f64::from(bf16(sink)?)
                } else {
                    f64::from(sink)
                };
                raw.extend(partial.finish(sink).map_err(|e| format!("{e:?}"))?);
            }
        }
        self.record("attention", 4096, raw)
    }

    fn add(&mut self, name: &'static str, x: &[f64], y: &[f64]) -> Result<Vec<f64>, String> {
        if x.len() != y.len() {
            return Err("RCA residual geometry differs".into());
        }
        self.record(name, 2048, x.iter().zip(y).map(|(&x, &y)| x + y).collect())
    }
}

fn dense(layer: &RealLayer<'_>, input: &[f32], scheme: Scheme) -> Result<Trace, String> {
    let rows = input.len() / 2048;
    super::super::validate(layer, input, rows)?;
    super::super::super::validate_expert(&layer.ffn.dense, 2048)?;
    if !matches!(layer.kind, Kind::Dense)
        || layer.sliding
        || input.iter().any(|v| v.to_bits() & 0xffff != 0)
    {
        return Err("RCA requires shared BF16 input and a dense NoPE layer".into());
    }
    let mut trace = Trace {
        scheme,
        rows,
        boundaries: Vec::new(),
    };
    let input = trace.record("input", 2048, input.iter().map(|&v| f64::from(v)).collect())?;
    let norm = trace.norm("pre_norm", &input, layer.pre_norm)?;
    let q = trace.project("q_proj", &norm, layer.q)?;
    let k = trace.project("k_proj", &norm, layer.k)?;
    let v = trace.project("v_proj", &norm, layer.v)?;
    let q = trace.norm("q_norm", &q, layer.q_norm)?;
    let k = trace.norm("k_norm", &k, layer.k_norm)?;
    let k = trace.cache("k_cache", &k)?;
    let v = trace.cache("v_cache", &v)?;
    let attended = trace.attention([&q, &k, &v], layer.sinks)?;
    let projected = trace.project("out_proj", &attended, layer.o)?;
    let residual = trace.add("residual", &input, &projected)?;
    let norm = trace.norm("post_norm", &residual, layer.post_norm)?;
    let gate = trace.project("gate", &norm, &layer.ffn.dense.gate)?;
    let up = trace.project("up", &norm, &layer.ffn.dense.up)?;
    let columns = layer.ffn.dense.gate.rows;
    let sig = trace.record(
        "sigmoid",
        columns,
        gate.iter().map(|g| 1.0 / (1.0 + (-g).exp())).collect(),
    )?;
    let silu = trace.record(
        "silu",
        columns,
        gate.iter().zip(sig).map(|(&g, p)| g * p).collect(),
    )?;
    let product = trace.record(
        "product",
        columns,
        silu.iter().zip(up).map(|(&s, u)| s * u).collect(),
    )?;
    let down = trace.project("down", &product, &layer.ffn.dense.down)?;
    trace.add("output", &residual, &down)?;
    Ok(trace)
}

#[test]
fn trace_publications_distinguish_frozen_two_rms_and_unrounded() {
    let value = 1.0 + 1.0 / 256.0;
    assert_eq!(Scheme::FrozenF16.publish(value).unwrap(), 1.0);
    assert_eq!(Scheme::Fp64.publish(value).unwrap(), value);
    assert!(Scheme::FrozenF16.publish(f64::NAN).is_err());
    assert!(Scheme::Fp64.publish(f64::INFINITY).is_err());
    let mut frozen = Trace {
        scheme: Scheme::FrozenF16,
        rows: 1,
        boundaries: vec![],
    };
    let mut two = Trace {
        scheme: Scheme::TwoRmsF16,
        rows: 1,
        boundaries: vec![],
    };
    let gamma = [1.0 + 3.0 / 256.0];
    assert_eq!(
        frozen.norm("norm", &[1.0], &gamma).unwrap(),
        vec![1.0078125]
    );
    assert_eq!(two.norm("norm", &[1.0], &gamma).unwrap(), vec![1.015625]);
    assert!(frozen.norm("norm", &[1.0], &[]).is_err());
    let mut cache = Trace {
        scheme: Scheme::FrozenF16,
        rows: 1,
        boundaries: vec![],
    };
    assert!(cache.cache("k_cache", &vec![1.0; 512]).is_ok());
    assert!(cache.cache("k_cache", &[]).is_err());
    assert!(cache.cache("k_cache", &vec![f64::MAX; 512]).is_err());
}

#[test]
#[ignore = "requires admitted CPU run and retained actual CUDA L0 boundaries; no qualification"]
fn replay_l0_p01_actual_operator_inputs() {
    let path = PathBuf::from(std::env::var("FIFI_FLINT_RCA_DEVICE_L0").unwrap());
    let bytes = std::fs::read(&path).unwrap();
    let sha = "88bcdf501fe655c749344bf669f507df3b2fe8edfd40b6e1f2c3b8e64cb480d8";
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), sha);
    let data: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(data["model_sha256"], MODEL_SHA256);
    assert_eq!(data["prefixes_sha256"], real_model::PREFIX_SHA256);
    assert_eq!(data["trace_matches_untraced"], true);
    assert_eq!(data["saved_m5_bit_exact"], true);
    assert_eq!(data["rows"], 5);
    let decode = |name: &str| {
        let stages = data["boundaries"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|stage| stage["name"] == name)
            .collect::<Vec<_>>();
        assert_eq!(stages.len(), 1);
        let stage = stages[0];
        assert_eq!(stage["source"], "CUDA");
        let bits: Vec<u32> = serde_json::from_value(stage["bits"].clone()).unwrap();
        assert_eq!(bits.len(), 5 * stage["columns"].as_u64().unwrap() as usize);
        assert!(
            bits.iter()
                .all(|&v| v & 0xffff == 0 && f32::from_bits(v).is_finite())
        );
        bits.into_iter()
            .map(|v| f64::from(f32::from_bits(v)))
            .collect::<Vec<_>>()
    };
    let root =
        real_model::fresh_directory("FIFI_FLINT_RCA_RESULT_DIR", std::slice::from_ref(&path));
    let (file, header, identity) = real_model::model();
    let layer = load(&file, &header, 0).unwrap();
    let layer = layer.view();
    let mut trace = Trace {
        scheme: Scheme::FrozenF16,
        rows: 5,
        boundaries: Vec::new(),
    };
    trace
        .norm("pre_norm", &decode("input"), layer.pre_norm)
        .unwrap();
    trace
        .norm("post_norm", &decode("residual"), layer.post_norm)
        .unwrap();
    for (name, input, weight) in [
        ("q_proj", "pre_norm", layer.q),
        ("k_proj", "pre_norm", layer.k),
        ("v_proj", "pre_norm", layer.v),
        ("out_proj", "attention", layer.o),
        ("gate", "post_norm", &layer.ffn.dense.gate),
        ("up", "post_norm", &layer.ffn.dense.up),
        ("down", "product", &layer.ffn.dense.down),
    ] {
        trace.project(name, &decode(input), weight).unwrap();
    }
    let gate = decode("gate");
    trace
        .record(
            "sigmoid",
            6144,
            gate.iter().map(|g| 1.0 / (1.0 + (-g).exp())).collect(),
        )
        .unwrap();
    for (name, left, right) in [("silu", "gate", "sigmoid"), ("product", "silu", "up")] {
        trace
            .record(
                name,
                6144,
                decode(left)
                    .iter()
                    .zip(decode(right))
                    .map(|(a, b)| a * b)
                    .collect(),
            )
            .unwrap();
    }
    trace
        .add("output", &decode("residual"), &decode("down"))
        .unwrap();
    let down = trace
        .boundaries
        .iter()
        .find(|stage| stage.name == "down")
        .unwrap()
        .values
        .clone();
    trace
        .add("down_replayed_output", &decode("residual"), &down)
        .unwrap();
    verify_prepared_identity(&file, &identity).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    real_model::write_new(
        &root.join("reference-l0-actual-inputs.json"),
        &serde_json::json!({
            "schema_version":1,"diagnostic_only":true,"qualification":false,"layer":0,"case":"p01","rows":5,
            "model_sha256":MODEL_SHA256,"prefixes_sha256":real_model::PREFIX_SHA256,
            "device_sha256":sha,"oracle_sha256":data["oracle_sha256"],"boundaries":trace.boundaries,
            "notes":"Each operator starts from its actual published CUDA input. F64 retains frozen decoded weights/scalars; no physical attention/cache replay, MMA reduction emulation, numerical floor, or model qualification."
        }),
    );
}

#[test]
#[ignore = "requires fifi-build and pinned GGUF/oracle; CPU only, diagnostic not qualification"]
fn capture_l0_p01_reference_boundaries() {
    let path = std::env::var("FIFI_FLINT_FULL_ORACLE").expect("frozen oracle");
    let sha = std::env::var("FIFI_FLINT_FULL_ORACLE_SHA256").expect("frozen oracle SHA256");
    let oracle = real_model::Oracle::parse(&std::fs::read(path).unwrap(), &sha).unwrap();
    let (file, header, identity) = real_model::model();
    let layer = load(&file, &header, 0).unwrap();
    let input = oracle.layers[0][0]
        .input_bits
        .iter()
        .map(|&v| f32::from_bits(v))
        .collect::<Vec<_>>();
    let frozen = dense(&layer.view(), &input, Scheme::FrozenF16).unwrap();
    let bits = frozen
        .boundaries
        .last()
        .unwrap()
        .values
        .iter()
        .map(|&v| (v as f32).to_bits())
        .collect::<Vec<_>>();
    // The new instrumentation is unusable unless it reproduces the pre-existing immutable oracle exactly.
    assert_eq!(
        bits, oracle.layers[0][0].output_bits,
        "RCA trace changed the frozen reference definition"
    );
    let schemes = [
        frozen,
        dense(&layer.view(), &input, Scheme::TwoRmsF16).unwrap(),
        dense(&layer.view(), &input, Scheme::RecBoundariesBf16).unwrap(),
        dense(&layer.view(), &input, Scheme::Fp64).unwrap(),
    ];
    verify_prepared_identity(&file, &identity).unwrap();
    let output =
        PathBuf::from(std::env::var("FIFI_FLINT_RCA_RESULT_DIR").expect("fresh RCA directory"));
    real_model::write_new(
        &output.join("reference-l0-p01.json"),
        &serde_json::json!({
            "schema_version":1,"layer":0,"case":"p01","rows":5,"diagnostic_only":true,
            "model_sha256":MODEL_SHA256,"oracle_sha256":sha,"prefixes_sha256":real_model::PREFIX_SHA256,
            "input_bits":oracle.layers[0][0].input_bits,"frozen_final_bit_exact":true,"schemes":schemes,
            "notes":"FP64 retains frozen decoded weights and F32 scalar coefficients, removes all intermediate BF16/F16 rounds. Rec-boundary mode changes RMS publications, norm/sink BF16 weights, and KV BF16 only: not Rec matmul/exp reduction order or publisher weight format. raw/values deltas are local publication errors, not an authorized gate or irreducible floor."
        }),
    );
}
