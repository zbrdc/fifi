//! One projection only, not a model-support or numerical-parity gate.
//! All shard headers are checked; only the three named payloads are acquired.
//! Bits and group size come from stored widths, not the repository's oQ6 name.

use std::{path::Path, process::ExitCode};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod projection {
    use std::{fs::File, io::Read, path::Path};

    use inference_engine::{
        kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec},
        tensor::{Dtype, Tensor},
        weights::{
            AcquireRequest, CancelHandle, Completion, Priority, Selection, ShardIo,
            StreamedCheckpoint,
        },
    };
    use serde_json::{Value, json};

    const BASE: &str = "language_model.model.layers.0.mlp.gate_proj";
    const SHARD: &str = "model-00001-of-00005.safetensors";
    const ROWS: u32 = 17_408;
    const WIDTH: u32 = 5_120;
    // Same header budget as qualification/q36/check-real.
    const MAX_HEADER_BYTES: u64 = 1 << 20;
    const INPUT_SEED: u32 = 0x5143_3806;

    fn keys() -> [String; 3] {
        ["weight", "scales", "biases"].map(|suffix| format!("{BASE}.{suffix}"))
    }

    /// Standard safetensors headers need not declare quantization parameters.
    /// Preserve explicit fields without interpreting undocumented metadata encodings.
    fn declarations(file: &mut File) -> Result<Value, String> {
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|e| format!("{SHARD} prefix: {e}"))?;
        let len = u64::from_le_bytes(prefix);
        if len > MAX_HEADER_BYTES {
            return Err(format!("{SHARD}: header exceeds {MAX_HEADER_BYTES} bytes"));
        }
        let mut bytes = vec![0; usize::try_from(len).map_err(|e| e.to_string())?];
        file.read_exact(&mut bytes)
            .map_err(|e| format!("{SHARD} header: {e}"))?;
        let header: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let entries: Vec<Value> = keys()
            .iter()
            .map(|key| {
                json!({
                    "key": key,
                    "bits": header[key]["bits"],
                    "group_size": header[key]["group_size"]
                })
            })
            .collect();
        Ok(json!({
            "shard_metadata": {
                "bits": header["__metadata__"]["bits"],
                "group_size": header["__metadata__"]["group_size"]
            },
            "entries": entries,
            "null_means": "field absent; no undocumented metadata encoding interpreted"
        }))
    }

    fn open(dir: &Path) -> Result<StreamedCheckpoint, String> {
        let index = std::fs::read(dir.join("model.safetensors.index.json"))
            .map_err(|e| format!("index: {e}"))?;
        let raw: Value = serde_json::from_slice(&index).map_err(|e| format!("index: {e}"))?;
        for key in keys() {
            if raw["weight_map"][&key].as_str() != Some(SHARD) {
                return Err(format!("{key}: index must name {SHARD}"));
            }
        }
        let mut shards: Vec<(String, Box<dyn ShardIo + Send>, u64)> = Vec::new();
        let mut declared = Value::Null;
        for number in 1..=5 {
            let name = format!("model-{number:05}-of-00005.safetensors");
            let mut file = File::open(dir.join(&name)).map_err(|e| format!("{name}: {e}"))?;
            let metadata = file.metadata().map_err(|e| format!("{name}: {e}"))?;
            if !metadata.is_file() {
                return Err(format!("{name}: shard must be a regular file"));
            }
            if name == SHARD {
                declared = declarations(&mut file)?;
            }
            shards.push((name, Box::new(file), metadata.len()));
        }
        // open seeks each reader back to zero and validates duplicate names,
        // stored extents, complete shard coverage, and index/header agreement.
        let checkpoint = StreamedCheckpoint::open(&index, shards, MAX_HEADER_BYTES)
            .map_err(|e| format!("StreamedCheckpoint::open: {e:?}"))?;
        println!(
            "oq6_checkpoint={}",
            json!({"index_headers_valid": true, "shards": 5, "payload_keys": keys()})
        );
        println!("oq6_header_declarations={declared}");
        Ok(checkpoint)
    }

    fn read(checkpoint: &StreamedCheckpoint, key: &str) -> Result<Tensor, String> {
        let (dtype, shape) = checkpoint
            .stored(key)
            .ok_or_else(|| format!("{key}: missing header entry"))?;
        let request = AcquireRequest::new(
            key.to_owned(),
            Selection::Whole,
            Priority::Demand,
            CancelHandle::new(),
        );
        let ticket = checkpoint
            .acquire(request)
            .map_err(|e| format!("{key}: acquire: {e:?}"))?;
        let tensor = match checkpoint.wait(ticket).map_err(|e| format!("{key}: {e:?}"))? {
            Completion::Ready(tensor) => tensor,
            Completion::Missing => return Err(format!("{key}: missing payload")),
        };
        let bytes = tensor.host_bytes().ok_or("tensor has no host bytes")?;
        if tensor.dtype() != dtype
            || tensor.shape() != shape
            || u64::try_from(bytes.len()).ok() != Some(tensor.byte_len())
        {
            return Err(format!("{key}: payload differs from checked header"));
        }
        // The acquisition path calls Tensor::try_from_host with the header's
        // dtype and shape, checking bytes == product(shape) * dtype byte width.
        println!(
            "oq6_tensor={}",
            json!({
                "key": key, "shard": SHARD, "dtype": format!("{dtype:?}"),
                "shape": shape, "byte_len": tensor.byte_len(),
                "byte_length_validation": "ShardHeader::read + Tensor::try_from_host"
            })
        );
        Ok(tensor)
    }

    fn quantization(payload: &Tensor, scales: &Tensor) -> Result<QuantizationSpec, String> {
        let &[rows, packed_width] = payload.shape() else {
            return Err("payload must have rank two".into());
        };
        let &[scale_rows, groups] = scales.shape() else {
            return Err("scales must have rank two".into());
        };
        if payload.dtype() != Dtype::U32 || rows != ROWS || scale_rows != ROWS || groups == 0 {
            return Err("expected U32 payload and scale rows matching logical shape".into());
        }
        let packed_bits = packed_width
            .checked_mul(32)
            .ok_or("packed bit count overflows u32")?;
        if !packed_bits.is_multiple_of(WIDTH) || !WIDTH.is_multiple_of(groups) {
            return Err("stored widths do not imply integral bits and group size".into());
        }
        let quantization = QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: packed_bits / WIDTH,
            group_size: WIDTH / groups,
        };
        println!(
            "oq6_quantization={}",
            json!({
                "mode": "affine", "bits": quantization.bits,
                "group_size": quantization.group_size, "logical_shape": [ROWS, WIDTH],
                "source": "header widths: packed_width*32/5120 and 5120/scales_width",
                "nominal_bits": 6, "nominal_bits_match": quantization.bits == 6
            })
        );
        Ok(quantization)
    }

    fn input() -> Result<Tensor, String> {
        // Xorshift32; integer fractions keep the input deterministic and bounded.
        let mut state = INPUT_SEED;
        let mut bytes = Vec::new();
        for _ in 0..WIDTH {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let value = ((state >> 16) as f32 + 0.5) / 32_768.0 - 1.0;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Tensor::try_from_host(Dtype::F32, vec![1, WIDTH], bytes).map_err(|e| format!("input: {e:?}"))
    }

    fn values(tensor: &Tensor, shape: &[u32]) -> Result<Vec<f32>, String> {
        if tensor.dtype() != Dtype::F32 || tensor.shape() != shape {
            return Err(format!(
                "expected F32 {shape:?}, received {:?} {:?}",
                tensor.dtype(),
                tensor.shape()
            ));
        }
        let bytes = tensor.host_bytes().ok_or("readback has no host bytes")?;
        let (chunks, remainder) = bytes.as_chunks::<4>();
        if !remainder.is_empty() {
            return Err("F32 readback has incomplete elements".into());
        }
        let values: Vec<f32> = chunks.iter().map(|b| f32::from_le_bytes(*b)).collect();
        if !values.iter().all(|v| v.is_finite()) {
            return Err("readback contains non-finite values".into());
        }
        Ok(values)
    }

    fn norm(values: &[f32]) -> f64 {
        values
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn compute(weight: &PackedWeight) -> Result<(), String> {
        let backend = |e| format!("MLX: {e:?}");
        let mut mlx = MlxKernels::try_new().map_err(backend)?;
        let packed = mlx.import_packed(weight).map_err(backend)?;
        let x = mlx.import_tensor(&input()?).map_err(backend)?;
        let y = mlx.qmm(&x, &packed).map_err(backend)?;
        let qmm = values(&mlx.readback(&y).map_err(backend)?, &[1, ROWS])?;
        let dense = mlx.dequantize(&packed).map_err(backend)?;
        let dense_host = mlx.readback(&dense).map_err(backend)?;
        let dequantized = values(&dense_host, &[ROWS, WIDTH])?;
        println!(
            "oq6_dequantize={}",
            json!({
                "dtype": "F32", "shape": [ROWS, WIDTH], "finite": true,
                "norm_l2": norm(&dequantized)
            })
        );
        drop(dequantized);
        drop(dense_host);
        let transposed = mlx.transpose(&dense, &[1, 0]).map_err(backend)?;
        let reference = mlx.mm(&x, &transposed).map_err(backend)?;
        let mm = values(&mlx.readback(&reference).map_err(backend)?, &[1, ROWS])?;
        let qmm_norm = norm(&qmm);
        let mm_norm = norm(&mm);
        let difference = qmm
            .iter()
            .zip(&mm)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>()
            .sqrt();
        // A zero reference has no defined relative error; report null, not a
        // fabricated denominator. No tolerance determines this run's exit code.
        let relative = (mm_norm > 0.0).then(|| difference / mm_norm);
        println!(
            "oq6_mm={}",
            json!({"dtype": "F32", "shape": [1, ROWS], "finite": true, "norm_l2": mm_norm})
        );
        println!(
            "oq6_projection={}",
            json!({
                "qmm_finite": true, "out_len": qmm.len(), "shape": [1, ROWS],
                "qmm_norm_l2": qmm_norm, "mm_norm_l2": mm_norm,
                "difference_norm_l2": difference, "relative_l2": relative,
                "relative_l2_definition": "norm(qmm-mm)/norm(mm); null if norm(mm)=0",
                "input_seed": INPUT_SEED, "input_shape": [1, WIDTH], "input_dtype": "F32",
                "bits": weight.quantization().bits, "group_size": weight.quantization().group_size,
                "proposed_tolerance": 1e-4, "tolerance_label": "provisional_empirical",
                "tolerance_reason": "F32 paths use different dequantization and reduction orders; measure on Mac",
                "parity_gate": false, "scope": BASE
            })
        );
        Ok(())
    }

    pub fn run(dir: &Path) -> Result<(), String> {
        let checkpoint = open(dir)?;
        let [payload_key, scales_key, biases_key] = keys();
        let payload = read(&checkpoint, &payload_key)?;
        let scales = read(&checkpoint, &scales_key)?;
        let biases = read(&checkpoint, &biases_key)?;
        let quantization = quantization(&payload, &scales)?;
        let result = PackedWeight::try_new(
            payload,
            scales,
            Some(biases),
            vec![ROWS, WIDTH],
            quantization,
        );
        match result {
            Ok(weight) => {
                println!("oq6_packed={}", json!({"accepted": true}));
                compute(&weight)
            }
            Err(error) => {
                println!(
                    "oq6_packed={}",
                    json!({"accepted": false, "error": format!("{error:?}")})
                );
                Err(format!(
                    "PackedWeight::try_new rejected header layout: {error:?}"
                ))
            }
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn run(dir: &Path) -> Result<(), String> {
    projection::run(dir)
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn run(_dir: &Path) -> Result<(), String> {
    Err("oq6-projection requires Apple Silicon macOS and native-mlx".into())
}

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let Some(dir) = args.next() else {
        eprintln!("usage: oq6-projection CHECKPOINT_DIR");
        return ExitCode::from(2);
    };
    if dir.is_empty() || args.next().is_some() {
        eprintln!("usage: oq6-projection CHECKPOINT_DIR");
        return ExitCode::from(2);
    }
    match run(Path::new(&dir)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            println!("oq6_error={}", serde_json::json!({"error": error}));
            ExitCode::from(1)
        }
    }
}
