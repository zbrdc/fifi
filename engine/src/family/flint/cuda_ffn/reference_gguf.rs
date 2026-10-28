//! Pin-bound CPU artifact preparation using existing Flint source ranges and GGUF decoding.
//! This does not broaden the generic Qwen GGUF reader's supported-type whitelist.

use std::{
    fs::File,
    io::{BufReader, Read, Seek, Write},
    ops::Range,
    os::unix::fs::{FileExt, MetadataExt},
    path::PathBuf,
};

use sha2::{Digest, Sha256};

#[cfg(test)]
#[path = "reference_chain.rs"]
mod chain;
#[cfg(test)]
#[path = "reference_full.rs"]
mod full;
#[cfg(test)]
#[path = "reference_rca.rs"]
mod rca;
#[cfg(test)]
#[path = "reference_routes.rs"]
pub(in crate::family::flint::cuda_ffn) mod routes;

use super::super::{Expert, Fixture, Kind, Matrix, RealLayer, validate_matrix};
use crate::{
    family::flint::{
        config::FULL_LAYERS,
        gguf_weights::{self, ExpertSources, MatrixSource, Nvfp4View},
    },
    weights::{GgufHeader, GgufTensor, GgufType, dequantize, repack_nvfp4_into},
};

pub(in crate::family::flint::cuda_ffn) const MODEL_SHA256: &str =
    "97cc8009ff9b132d5cf3c488e6767962bb5a5dc158a90ed661bbbddf876b8bde";

struct Layer {
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    pre_norm: Vec<f32>,
    post_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    sinks: Vec<f32>,
    ffn: Fixture,
    kind: Kind,
    sliding: bool,
}

impl Layer {
    fn view(&self) -> RealLayer<'_> {
        RealLayer {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            o: &self.o,
            pre_norm: &self.pre_norm,
            post_norm: &self.post_norm,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            sinks: &self.sinks,
            ffn: &self.ffn,
            kind: self.kind,
            sliding: self.sliding,
        }
    }
}

pub(in crate::family::flint::cuda_ffn) fn raw(
    file: &File,
    range: Range<u64>,
) -> Result<Vec<u8>, String> {
    let file_len = file.metadata().map_err(|e| format!("{e:?}"))?.len();
    let length = range
        .end
        .checked_sub(range.start)
        .filter(|_| range.end <= file_len)
        .and_then(|length| usize::try_from(length).ok())
        .ok_or_else(|| "invalid source byte range".to_string())?;
    let mut data = Vec::new();
    data.try_reserve_exact(length)
        .map_err(|e| format!("{e:?}"))?;
    data.resize(length, 0);
    file.read_exact_at(&mut data, range.start)
        .map_err(|e| format!("{e:?}"))?;
    if file.metadata().map_err(|e| format!("{e:?}"))?.len() != file_len {
        return Err("source file length changed during read".into());
    }
    Ok(data)
}

fn decode(view: Nvfp4View<'_>) -> Result<Matrix, String> {
    let matrix = Matrix {
        rows: view.shape[0],
        cols: view.shape[1],
        values: dequantize(GgufType::Nvfp4, view.data).map_err(|e| format!("{e:?}"))?,
        factor: view.tensor_scale,
    };
    validate_matrix(&matrix)?;
    let mut packed = vec![0; matrix.values.len() / 2];
    let mut scales = vec![0; matrix.values.len() / 16];
    // Qualification only: expected values never come from the backend re-layout.
    repack_nvfp4_into(view.data, &mut packed, &mut scales).map_err(|e| format!("{e:?}"))?;
    Ok(matrix)
}

fn matrix(file: &File, source: &MatrixSource) -> Result<Matrix, String> {
    let data = raw(file, source.data.clone())?;
    let scale = raw(file, source.scale.clone())?;
    decode(source.bind(&data, &scale).map_err(|e| format!("{e:?}"))?)
}

pub(in crate::family::flint::cuda_ffn) fn plain(
    file: &File,
    tensor: &GgufTensor,
) -> Result<Vec<f32>, String> {
    dequantize(tensor.ggml_type(), &raw(file, tensor.abs_range())?).map_err(|e| format!("{e:?}"))
}

fn expert(file: &File, source: ExpertSources) -> Result<Expert, String> {
    Ok(Expert {
        gate: matrix(file, &source.gate)?,
        up: matrix(file, &source.up)?,
        down: matrix(file, &source.down)?,
    })
}

pub(in crate::family::flint::cuda_ffn) fn scalar_sources(
    header: &GgufHeader,
    names: [&str; 3],
    intermediate: usize,
) -> Result<ExpertSources, String> {
    let source = |name: &str, shape| {
        gguf_weights::matrix_source(header, name, shape).map_err(|e| format!("{e:?}"))
    };
    Ok(ExpertSources {
        gate: source(names[0], [intermediate, 2048])?,
        up: source(names[1], [intermediate, 2048])?,
        down: source(names[2], [2048, intermediate])?,
    })
}

fn unused_expert() -> Expert {
    let empty = Matrix {
        rows: 0,
        cols: 0,
        values: vec![],
        factor: 1.0,
    };
    Expert {
        gate: empty.clone(),
        up: empty.clone(),
        down: empty,
    }
}

fn ffn(file: &File, header: &GgufHeader, layer: u32) -> Result<Fixture, String> {
    let unused = unused_expert();
    let (kind, _) = layer_topology(layer)?;
    if matches!(kind, Kind::Dense) {
        let [gate, up, down] =
            ["ffn_gate", "ffn_up", "ffn_down"].map(|name| format!("blk.{layer}.{name}"));
        let dense = expert(file, scalar_sources(header, [&gate, &up, &down], 6144)?)?;
        return Ok(Fixture {
            hidden: 2048,
            router: unused.gate.clone(),
            experts: vec![],
            shared: unused,
            dense,
        });
    }
    let router = gguf_weights::required(
        header,
        &format!("blk.{layer}.ffn_gate_inp.weight"),
        GgufType::Bf16,
        &[2048, 256],
    )
    .map_err(|e| format!("{e:?}"))?;
    let [gate, up, down] = ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"]
        .map(|name| format!("blk.{layer}.{name}"));
    let shared = expert(file, scalar_sources(header, [&gate, &up, &down], 1024)?)?;
    let experts = (0..256)
        .map(|index| {
            let source =
                gguf_weights::expert_sources(header, layer, index).map_err(|e| format!("{e:?}"))?;
            expert(file, source)
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Fixture {
        hidden: 2048,
        router: Matrix {
            rows: 256,
            cols: 2048,
            values: plain(file, router)?,
            factor: 1.0,
        },
        experts,
        shared,
        dense: unused,
    })
}

fn layer_topology(layer: u32) -> Result<(Kind, bool), String> {
    if layer >= 49 {
        return Err("pinned Flint layer index must be less than 49".into());
    }
    Ok((
        if layer < 2 { Kind::Dense } else { Kind::Moe },
        !FULL_LAYERS.contains(&u64::from(layer)),
    ))
}

fn load(file: &File, header: &GgufHeader, layer: u32) -> Result<Layer, String> {
    let (kind, sliding) = layer_topology(layer)?;
    let projection = |name, shape| {
        let source = gguf_weights::matrix_source(header, &format!("blk.{layer}.{name}"), shape)
            .map_err(|e| format!("{e:?}"))?;
        matrix(file, &source)
    };
    let vector = |name, width| {
        let tensor = gguf_weights::required(
            header,
            &format!("blk.{layer}.{name}.weight"),
            GgufType::F32,
            &[width],
        )
        .map_err(|e| format!("{e:?}"))?;
        plain(file, tensor)
    };
    Ok(Layer {
        q: projection("attn_q", [4096, 2048])?,
        k: projection("attn_k", [512, 2048])?,
        v: projection("attn_v", [512, 2048])?,
        o: projection("attn_output", [2048, 4096])?,
        pre_norm: vector("attn_norm", 2048)?,
        post_norm: vector("ffn_norm", 2048)?,
        q_norm: vector("attn_q_norm", 128)?,
        k_norm: vector("attn_k_norm", 128)?,
        sinks: vector("attn_sinks", 32)?,
        ffn: ffn(file, header, layer)?,
        kind,
        sliding,
    })
}

fn sha256(file: &File) -> Result<String, String> {
    let mut reader = file.try_clone().map_err(|e| format!("{e:?}"))?;
    reader.rewind().map_err(|e| format!("{e:?}"))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    loop {
        let count = reader.read(&mut buffer).map_err(|e| format!("{e:?}"))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(in crate::family::flint::cuda_ffn) fn verify_prepared_identity(
    file: &File,
    identity: &serde_json::Value,
) -> Result<(), String> {
    let meta = file.metadata().map_err(|e| format!("{e:?}"))?;
    let expected = serde_json::json!({
        "schema_version": 1, "model_sha256": MODEL_SHA256,
        "size": meta.len(), "device": meta.dev(), "inode": meta.ino(),
        "mtime_seconds": meta.mtime(), "mtime_nanoseconds": meta.mtime_nsec(),
        "ctime_seconds": meta.ctime(), "ctime_nanoseconds": meta.ctime_nsec(),
    });
    for (field, value) in expected.as_object().unwrap() {
        if identity.get(field) != Some(value) {
            return Err(format!("prepared model identity mismatch: {field}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> MatrixSource {
        MatrixSource {
            data: 0..36,
            scale: 36..40,
            shape: [1, 64],
        }
    }

    fn payload() -> Vec<u8> {
        let mut data = vec![0x22; 36];
        data[..4].fill(0x38);
        data
    }

    #[test]
    fn nvfp4_matrix_keeps_the_tensor_multiplier_separate() {
        let data = payload();
        let source = source();
        let view = source.bind(&data, &2.75_f32.to_le_bytes()).unwrap();
        let matrix = decode(view).unwrap();
        assert_eq!((matrix.rows, matrix.cols), (1, 64));
        assert_eq!(matrix.values, vec![1.0; 64]);
        assert_eq!(matrix.factor.to_bits(), 2.75_f32.to_bits());
    }

    #[test]
    fn decoding_refuses_nonpositive_factors_and_wrong_shape() {
        let data = payload();
        let mut source = source();
        for factor in [0.0_f32, -1.0] {
            assert!(decode(source.bind(&data, &factor.to_le_bytes()).unwrap()).is_err());
        }
        source.shape = [2, 64];
        assert!(decode(source.bind(&data, &1.0_f32.to_le_bytes()).unwrap()).is_err());
    }

    #[test]
    fn decoding_refuses_groups_the_gpu_cannot_represent() {
        let mut data = payload();
        data[0] = 0xff;
        let source = source();
        assert!(decode(source.bind(&data, &1.0_f32.to_le_bytes()).unwrap()).is_err());
    }

    #[test]
    fn positioned_reads_refuse_bad_ranges_and_truncation() {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(&[1, 2, 3, 4]).unwrap();
        assert_eq!(
            sha256(&file).unwrap(),
            format!("{:x}", Sha256::digest([1_u8, 2, 3, 4]))
        );
        assert_eq!(raw(&file, 0..4).unwrap(), vec![1, 2, 3, 4]);
        assert!(raw(&file, Range { start: 3, end: 2 }).is_err());
        assert!(raw(&file, 0..u64::MAX).is_err());
        file.set_len(3).unwrap();
        assert!(raw(&file, 0..4).is_err());
    }

    fn identity_receipt(file: &File) -> serde_json::Value {
        let meta = file.metadata().unwrap();
        serde_json::json!({
            "schema_version": 1, "model_sha256": MODEL_SHA256,
            "size": meta.len(), "device": meta.dev(), "inode": meta.ino(),
            "mtime_seconds": meta.mtime(), "mtime_nanoseconds": meta.mtime_nsec(),
            "ctime_seconds": meta.ctime(), "ctime_nanoseconds": meta.ctime_nsec(),
        })
    }

    #[test]
    fn model_identity_rejects_changed_receipts() {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(&[1, 2, 3, 4]).unwrap();
        let identity = identity_receipt(&file);
        verify_prepared_identity(&file, &identity).unwrap();
        for (key, value) in identity.as_object().unwrap() {
            let mut changed = identity.clone();
            changed[key] = value
                .as_u64()
                .map_or_else(|| serde_json::json!("wrong"), |n| serde_json::json!(n + 1));
            assert!(verify_prepared_identity(&file, &changed).is_err(), "{key}");
            changed[key] = serde_json::Value::Null;
            assert!(verify_prepared_identity(&file, &changed).is_err(), "{key}");
            changed.as_object_mut().unwrap().remove(key);
            assert!(verify_prepared_identity(&file, &changed).is_err(), "{key}");
        }
        for invalid in [serde_json::json!([]), serde_json::json!(null)] {
            assert!(verify_prepared_identity(&file, &invalid).is_err());
        }
    }

    #[test]
    fn model_identity_rechecks_open_file_metadata() {
        let file = tempfile::tempfile().unwrap();
        let identity = identity_receipt(&file);
        verify_prepared_identity(&file, &identity).unwrap();
        let modified = file.metadata().unwrap().modified().unwrap();
        file.set_times(
            std::fs::FileTimes::new().set_modified(modified + std::time::Duration::from_secs(1)),
        )
        .unwrap();
        assert!(verify_prepared_identity(&file, &identity).is_err());
    }

    #[test]
    #[ignore = "requires explicit CPU model-preparation GO and the pinned Flint GGUF"]
    fn freeze_pinned_gguf_layers_0_and_2() {
        let path = PathBuf::from(std::env::var("FIFI_FLINT_GGUF").expect("pinned GGUF path"));
        let output =
            PathBuf::from(std::env::var("FIFI_FLINT_ORACLE_DIR").expect("fresh oracle directory"));
        assert!(output.is_dir());
        let identity_path = PathBuf::from(
            std::env::var("FIFI_FLINT_MODEL_IDENTITY").expect("host model identity receipt"),
        );
        let identity: serde_json::Value =
            serde_json::from_reader(File::open(identity_path).unwrap()).unwrap();
        let mut file = File::open(&path).unwrap();
        verify_prepared_identity(&file, &identity).unwrap();
        file.rewind().unwrap();
        let header = GgufHeader::parse(
            &mut BufReader::new(&file),
            file.metadata().unwrap().len(),
            64 << 20,
        )
        .unwrap();
        let tokens = [0, 1, 2, 3];
        let mut input = Vec::new();
        for token in tokens {
            let range = gguf_weights::embedding_row(&header, token).unwrap();
            input.extend(dequantize(GgufType::Bf16, &raw(&file, range).unwrap()).unwrap());
        }
        let mut artifacts = Vec::new();
        for index in [0, 2] {
            let layer = load(&file, &header, index).unwrap();
            let expected = super::super::run(&layer.view(), &input, tokens.len()).unwrap();
            artifacts.push((index, serde_json::json!({
                "schema_version": 1, "model_sha256": MODEL_SHA256,
                "scope": "independent layer from fixed embeddings, not a full model or text prompt",
                "layer": index, "rows": tokens.len(), "physical_kv": "F16",
                "sliding": layer.sliding,
                "arithmetic": "FP64 projections, normalization, RoPE, attention and FFN; explicit BF16 publications",
                "token_ids": tokens, "input_bits": input.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "output_bits": expected.output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "ids": expected.ids, "gates_bits": expected.gates.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "sorted_pairs": expected.sorted_pairs, "inverse": expected.inverse,
                "max_intermediate_bits": expected.max_intermediate.to_bits(),
            })));
        }
        verify_prepared_identity(&file, &identity).unwrap();
        for (index, artifact) in artifacts {
            let path = output.join(format!("layer-{index}-expected.json"));
            let mut target = File::options()
                .write(true)
                .create_new(true)
                .open(path)
                .unwrap();
            target
                .write_all(&serde_json::to_vec(&artifact).unwrap())
                .unwrap();
            target.sync_all().unwrap();
        }
    }
}
