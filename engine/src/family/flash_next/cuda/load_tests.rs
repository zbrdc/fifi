//! Tests of the trunk loader.
//!
//! - Host, no pack: the form rules and the layer grouping on synthetic visits.
//! - Ignored, headers of the real pack in `FIFI_FLASH_NEXT_EXL3_PACK` (the full local pack or a
//!   sparse copy of it): the list is the pack's index less the routed experts, the embedding, the
//!   MTP head and the vision tower, and its bytes are the walk's trunk bytes.
//! - Ignored, device (native-cuda): layer 3 uploads under its pack names and reads back the bytes
//!   the host reads from the same file. Needs a GPU and about 60 MiB, and the payload of layer 3.

use std::collections::BTreeSet;
use std::path::Path;

use super::load::{Form, TrunkList};
use crate::{
    EngError,
    family::exl3_pack::{EMBED_NAME, Visit, VisitKind},
    tensor::Dtype,
    weights::exl3::Checkpoint,
};

const PREFIX: &str = "model.language_model.layers";

fn plain(name: &str, keep: bool, bytes: u64) -> Visit {
    Visit {
        name: name.to_owned(),
        kind: VisitKind::Plain {
            dtype: Dtype::F16,
            shape: vec![4],
            keep,
        },
        bytes,
    }
}

fn host(name: &str) -> Visit {
    Visit {
        name: name.to_owned(),
        kind: VisitKind::Host {
            dtype: Dtype::F16,
            shape: vec![4],
        },
        bytes: 0,
    }
}

fn matrix(name: &str, bytes: u64) -> Visit {
    Visit {
        name: name.to_owned(),
        kind: VisitKind::Matrix {
            k: 128,
            n: 128,
            rate: 5,
        },
        bytes,
    }
}

fn synthetic() -> Vec<Visit> {
    let at = |suffix: &str| format!("{PREFIX}.2.{suffix}");
    vec![
        plain(
            "model.language_model.hyper_connection_mixer.hc_norm.weight",
            true,
            8,
        ),
        matrix("lm_head", 100),
        plain(EMBED_NAME, true, 1000),
        plain(
            &at("attn_hyper_connection.block_inject_weight.weight"),
            true,
            8,
        ),
        plain(&at("linear_attn.in_proj_a.weight"), false, 16),
        plain(&at("linear_attn.A_log"), false, 4),
        plain(&at("linear_attn.norm.weight"), false, 4),
        plain(&at("self_attn.q_norm.weight"), false, 4),
        plain(&at("self_attn.indexer.k_layernorm.weight"), false, 4),
        host(&at("ple.norm_key.weight")),
        host(&at("ple.conv1d.weight")),
        plain(&at("ple.key_proj.weight"), true, 8),
        plain(&at("mlp.gate.weight"), false, 4),
        matrix(&at("mlp.experts.7.up_proj"), 5000),
        matrix(&at("mlp.shared_expert.up_proj"), 50),
    ]
}

#[test]
fn the_form_follows_the_walks_rule_and_the_role_of_the_name() {
    let list = TrunkList::from_visits(&synthetic(), 4).unwrap();
    let form = |suffix: &str| {
        list.all()
            .find(|t| t.name.ends_with(suffix))
            .unwrap_or_else(|| panic!("{suffix} is not in the trunk"))
            .form
    };
    assert_eq!(form("hc_norm.weight"), Form::Half);
    assert_eq!(form("lm_head"), Form::Matrix);
    assert_eq!(form("block_inject_weight.weight"), Form::Half);
    assert_eq!(form("in_proj_a.weight"), Form::Float, "the gates are F32");
    assert_eq!(form("linear_attn.A_log"), Form::Decay);
    assert_eq!(
        form("linear_attn.norm.weight"),
        Form::Float,
        "the gated norm is not rms0"
    );
    assert_eq!(
        form("self_attn.q_norm.weight"),
        Form::NormPlusOne,
        "attn_prep multiplies by it"
    );
    // qsa_index.cu applies (1.0f + w[d]) itself, so the stored weight goes up as it is.
    assert_eq!(
        form("indexer.k_layernorm.weight"),
        Form::Float,
        "qsa_index adds the 1"
    );
    assert_eq!(form("ple.key_proj.weight"), Form::Half);
    assert_eq!(form("mlp.gate.weight"), Form::Float);
    assert_eq!(form("shared_expert.up_proj"), Form::Matrix);
}

#[test]
fn experts_and_the_embedding_stay_out_and_the_rest_is_grouped_by_layer() {
    let list = TrunkList::from_visits(&synthetic(), 4).unwrap();
    assert!(
        list.all()
            .all(|t| !t.name.contains(".mlp.experts.") && t.name != EMBED_NAME)
    );
    assert_eq!(list.root().len(), 2);
    assert_eq!(list.layer(2).unwrap().len(), 9);
    assert!(list.layer(0).unwrap().is_empty());
    assert!(list.layer(4).is_err());
    // The root (mixer norm, head), then layer 2 in visit order.
    let expected: u64 = [8, 100, 8, 16, 4, 4, 4, 4, 8, 4, 50].iter().sum();
    assert_eq!(list.bytes(), expected);
}

#[test]
fn a_visit_under_a_layer_the_model_lacks_is_refused() {
    let visits = vec![plain(&format!("{PREFIX}.9.mlp.gate.weight"), false, 4)];
    assert!(matches!(
        TrunkList::from_visits(&visits, 4),
        Err(EngError::Checkpoint(_))
    ));
}

#[test]
fn a_tensor_without_a_role_is_refused_by_name_and_never_widened_by_default() {
    let at = |suffix: &str| format!("{PREFIX}.1.{suffix}");
    for (visit, named) in [
        (
            plain(&at("linear_attn.new_bias"), false, 4),
            "linear_attn.new_bias",
        ),
        (
            plain(&at("self_attn.indexer.q_norm.weight"), false, 4),
            "indexer.q_norm",
        ),
        (plain(&at("linear_attn.dt_bias"), true, 4), "dt_bias"),
        (matrix(&at("mlp.router_proj"), 4), "router_proj"),
    ] {
        let Err(EngError::Checkpoint(message)) = TrunkList::from_visits(&[visit], 4) else {
            panic!("{named} was accepted");
        };
        assert!(
            message.contains(named) && message.contains("no device form"),
            "{message}"
        );
    }
}

fn pack() -> Checkpoint {
    let dir = std::env::var_os("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK");
    Checkpoint::open(Path::new(&dir)).expect("open the pack")
}

/// The component names a matrix prefix expands to in the index.
const MATRIX_PARTS: [&str; 4] = ["trellis", "suh", "svh", "mul1"];

fn expanded(list: &TrunkList) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for tensor in list.all() {
        if tensor.form == Form::Matrix {
            for part in MATRIX_PARTS {
                assert!(
                    names.insert(format!("{}.{part}", tensor.name)),
                    "{}",
                    tensor.name
                );
            }
        } else {
            assert!(names.insert(tensor.name.clone()), "{}", tensor.name);
        }
    }
    names
}

#[test]
#[ignore = "reads the headers of the real pack in FIFI_FLASH_NEXT_EXL3_PACK"]
fn the_list_is_the_pack_without_experts_embedding_mtp_and_vision() {
    let checkpoint = pack();
    let list = TrunkList::new(&checkpoint.source, &checkpoint.model_config).expect("trunk list");
    assert_eq!(list.layers(), 48);
    let listed = expanded(&list);

    let dir = std::env::var_os("FIFI_FLASH_NEXT_EXL3_PACK").unwrap();
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(Path::new(&dir).join("model.safetensors.index.json")).expect("index"),
    )
    .expect("index json");
    let held: BTreeSet<String> = index["weight_map"]
        .as_object()
        .expect("weight_map")
        .keys()
        .filter(|name| {
            !name.starts_with("mtp.")
                && !name.starts_with("model.visual.")
                && !name.contains(".mlp.experts.")
                && !name.contains(".ple.norm_")
                && !name.contains(".ple.conv1d")
                && *name != EMBED_NAME
        })
        .cloned()
        .collect();
    let missing: Vec<_> = held.difference(&listed).take(5).collect();
    let extra: Vec<_> = listed.difference(&held).take(5).collect();
    assert!(missing.is_empty(), "pack names the list lacks: {missing:?}");
    assert!(extra.is_empty(), "list names the pack lacks: {extra:?}");
    assert_eq!(listed.len(), held.len());
}

#[test]
#[ignore = "reads the headers of the real pack in FIFI_FLASH_NEXT_EXL3_PACK"]
fn the_list_bytes_are_the_walks_trunk_bytes() {
    let checkpoint = pack();
    let (report, visits) = crate::family::flash_next::load::exl3_pack::inspect_flash_pack_listing(
        &checkpoint.source,
        &checkpoint.model_config,
    )
    .expect("walk");
    let list = TrunkList::new(&checkpoint.source, &checkpoint.model_config).expect("trunk list");
    let embedding = visits
        .iter()
        .find(|v| v.name == EMBED_NAME)
        .map(|v| v.bytes)
        .expect("embedding");
    // The host feed reads exactly the PLE norms and the convolution: four tensors, zero bytes.
    let host_only: Vec<&str> = visits
        .iter()
        .filter(|v| matches!(v.kind, VisitKind::Host { .. }))
        .map(|v| v.name.as_str())
        .collect();
    assert_eq!(host_only.len(), 4, "{host_only:?}");
    assert!(
        host_only
            .iter()
            .all(|n| n.contains(".ple.norm_") || n.contains(".ple.conv1d"))
    );
    // The plan's rule (FlashLoadPlan): device bytes less the experts and the embedding.
    assert_eq!(
        list.bytes(),
        report.pack.device_bytes - report.pack.expert_bytes - embedding
    );
    assert_eq!(embedding, 248_320 * 2560 * 2);
    let matrices = list.all().filter(|t| t.form == Form::Matrix).count();
    // 301 matrices at 5 bpw and the 12 indexers at 3 bpw are trunk; the rest are experts.
    assert_eq!(report.pack.matrices - matrices, 48 * 512 * 3);
    eprintln!(
        "trunk {} tensors, {} bytes",
        list.all().count(),
        list.bytes()
    );
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod device {
    use half::{bf16, f16};

    use super::super::load::{
        TrunkList, TrunkTensor,
        upload::{layer_weights, root_weights},
    };
    use super::pack;
    use crate::{
        family::{
            project::cuda::{CudaProjector, CudaWeight, CudaWeights, Reserve},
            qwen36::cuda::exl3_source::PackReader,
        },
        kernels::cuda::{CudaContext, DeviceBuffer, device_present},
        tensor::Dtype,
        weights::{WeightSource, exl3::Checkpoint},
    };

    /// FNV-1a over 8-byte words, then the tail bytes, with the length folded in.
    fn checksum(bytes: &[u8]) -> u64 {
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ bytes.len() as u64;
        let (words, tail) = bytes.as_chunks::<8>();
        for word in words {
            hash = (hash ^ u64::from_le_bytes(*word)).wrapping_mul(PRIME);
        }
        for &byte in tail {
            hash = (hash ^ u64::from(byte)).wrapping_mul(PRIME);
        }
        hash
    }

    fn device_checksum(context: &CudaContext, buffer: &DeviceBuffer) -> u64 {
        let mut bytes = vec![0u8; buffer.bytes()];
        context.download(buffer, &mut bytes).expect("download");
        checksum(&bytes)
    }

    fn f32_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
        values.flat_map(|v| v.to_le_bytes()).collect()
    }

    /// The stored values of a plain tensor as F32, widened by the `half` crate here and not by the
    /// loader's own helper.
    fn stored_f32(checkpoint: &Checkpoint, name: &str) -> Vec<f32> {
        let selected = checkpoint
            .source
            .read_selected(&[name], 1 << 30)
            .expect("read");
        let tensor = selected.tensor(name).expect("tensor");
        let bytes = tensor.host_bytes().expect("host bytes");
        match tensor.dtype() {
            Dtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|w| f16::from_bits(u16::from_le_bytes(*w)).to_f32())
                .collect(),
            Dtype::Bf16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|w| bf16::from_bits(u16::from_le_bytes(*w)).to_f32())
                .collect(),
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect(),
            other => panic!("{name}: {other:?}"),
        }
    }

    fn device_buffers<'w>(weights: &'w CudaWeights, name: &str) -> Vec<&'w DeviceBuffer> {
        match weights
            .get(name)
            .unwrap_or_else(|| panic!("{name} not loaded"))
        {
            CudaWeight::Exl3 { weight, .. } => vec![&weight.trellis, &weight.suh, &weight.svh],
            CudaWeight::Raw { buffer, .. } => vec![buffer],
            _ => panic!("{name}: unexpected weight kind"),
        }
    }

    /// The checksums of the bytes the device should hold for `tensor`, from the host side and from
    /// each consumer kernel's documented contract, not from the loader's form table:
    /// - `attn_prep` multiplies the RMS-normalized head by the weight it gets (`q_norm`, `k_norm`
    ///   arrive as `1 + w`);
    /// - `qsa_index.cu` adds the 1 itself (the indexer norms arrive as stored);
    /// - the scan kernel reads `-exp(A_log)`;
    /// - the hyper kernels and the PLE matvec read the stored half bytes;
    /// - an EXL3 matrix is its trellis as stored and its `suh`, `svh` widened to F32;
    /// - every other plain tensor is its stored values as F32.
    fn expected(
        reader: &PackReader<'_>,
        checkpoint: &Checkpoint,
        tensor: &TrunkTensor,
    ) -> Vec<u64> {
        let name = tensor.name.as_str();
        let stored_half = name.contains("hyper_connection")
            || name.ends_with("ple.key_proj.weight")
            || name.ends_with("ple.value_proj.weight");
        if !stored_half && checkpoint.source.contains(&format!("{name}.trellis")) {
            let matrix = reader.matrix(name).expect("matrix");
            let widened = |scales: &crate::tensor::Tensor| {
                f32_bytes(
                    scales
                        .host_bytes()
                        .expect("scales")
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|w| f16::from_bits(u16::from_le_bytes(*w)).to_f32()),
                )
            };
            return vec![
                checksum(matrix.trellis().host_bytes().expect("trellis")),
                checksum(&widened(matrix.suh())),
                checksum(&widened(matrix.svh())),
            ];
        }
        if stored_half {
            let selected = checkpoint
                .source
                .read_selected(&[name], 1 << 30)
                .expect("read");
            let tensor = selected.tensor(name).expect("tensor");
            return vec![checksum(tensor.host_bytes().expect("host bytes"))];
        }
        let stored = stored_f32(checkpoint, name);
        let values: Vec<f32> = if name.ends_with("self_attn.q_norm.weight")
            || name.ends_with("self_attn.k_norm.weight")
        {
            stored.into_iter().map(|w| w + 1.0).collect()
        } else if name.ends_with("linear_attn.A_log") {
            stored.into_iter().map(|a| -a.exp()).collect()
        } else {
            stored
        };
        vec![checksum(&f32_bytes(values.into_iter()))]
    }

    /// Uploads `tensors` and compares every device buffer with the expectation of its kernel's
    /// contract; the device bytes must equal the bytes the walk counted for them.
    fn check(
        projector: &CudaProjector,
        checkpoint: &Checkpoint,
        what: &str,
        tensors: &[TrunkTensor],
        upload: impl FnOnce(&PackReader<'_>) -> Vec<CudaWeight>,
    ) {
        let reader = PackReader::new(projector, checkpoint, Reserve::default());
        let loaded = upload(&reader);
        assert_eq!(loaded.len(), tensors.len(), "{what}: tensor count");
        let weights = CudaWeights::from_exl3(loaded).expect("weights");
        let context = projector.context();
        for tensor in tensors {
            let want = expected(&reader, checkpoint, tensor);
            let got: Vec<u64> = device_buffers(&weights, &tensor.name)
                .into_iter()
                .map(|buffer| device_checksum(context, buffer))
                .collect();
            assert_eq!(got, want, "{} ({:?})", tensor.name, tensor.form);
        }
        let counted: u64 = tensors.iter().map(|t| t.bytes).sum();
        assert_eq!(
            weights.device_bytes() as u64,
            counted,
            "{what}: device bytes against the walk"
        );
        eprintln!("{what}: {} tensors, {counted} device bytes", tensors.len());
    }

    fn open() -> (Checkpoint, TrunkList, CudaProjector) {
        assert!(
            device_present("Flash-Next trunk upload"),
            "a GPU is required"
        );
        let checkpoint = pack();
        let list = TrunkList::new(&checkpoint.source, &checkpoint.model_config).expect("list");
        (checkpoint, list, CudaProjector::new().expect("projector"))
    }

    fn has(tensors: &[TrunkTensor], suffix: &str) -> bool {
        tensors.iter().any(|t| t.name.ends_with(suffix))
    }

    #[test]
    #[ignore = "needs a GPU (about 60 MiB) and FIFI_FLASH_NEXT_EXL3_PACK with the payload of layer 3"]
    fn layer_three_uploads_under_its_pack_names_and_reads_back_the_host_bytes() {
        let (checkpoint, list, projector) = open();
        let layer = std::env::var("FIFI_FLASH_NEXT_EXL3_MOE_LAYER")
            .map_or(3, |value| value.parse().expect("a layer number"));
        let tensors = list.layer(layer).expect("layer");
        assert!(
            has(tensors, "self_attn.q_proj"),
            "layer {layer} is not a QSA layer"
        );
        assert!(
            has(tensors, "self_attn.q_norm.weight") && has(tensors, "indexer.q_layernorm.weight")
        );
        check(
            &projector,
            &checkpoint,
            &format!("layer {layer}"),
            tensors,
            |reader| layer_weights(reader, &list, layer).expect("upload layer"),
        );
    }

    /// Layer 1 is a DeltaNet layer and the PLE layer: `-exp(A_log)`, the F32 gates `in_proj_a` and
    /// `in_proj_b`, `dt_bias`, the convolution and the gated norm, and the two PLE projections as
    /// stored half. The PLE norms and convolution are host-only and must not be on the card.
    #[test]
    #[ignore = "needs a GPU (about 120 MiB) and the full pack in FIFI_FLASH_NEXT_EXL3_PACK"]
    fn the_deltanet_ple_layer_uploads_and_reads_back_the_host_bytes() {
        let (checkpoint, list, projector) = open();
        let tensors = list.layer(1).expect("layer");
        for suffix in [
            "linear_attn.A_log",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.dt_bias",
            "linear_attn.conv1d.weight",
            "linear_attn.norm.weight",
            "ple.key_proj.weight",
            "ple.value_proj.weight",
        ] {
            assert!(has(tensors, suffix), "{suffix} is missing from layer 1");
        }
        assert!(!has(tensors, "ple.norm_key.weight") && !has(tensors, "ple.conv1d.weight"));
        check(&projector, &checkpoint, "layer 1", tensors, |reader| {
            layer_weights(reader, &list, 1).expect("upload layer")
        });
    }

    /// The mixer that feeds `lm_head` (three stored half tensors) and `lm_head` itself.
    #[test]
    #[ignore = "needs a GPU (under 1 GiB) and the full pack in FIFI_FLASH_NEXT_EXL3_PACK"]
    fn the_root_tensors_upload_and_read_back_the_host_bytes() {
        let (checkpoint, list, projector) = open();
        let tensors = list.root();
        assert!(has(tensors, "lm_head") && has(tensors, "hyper_connection_mixer.hc_norm.weight"));
        check(&projector, &checkpoint, "root", tensors, |reader| {
            root_weights(reader, &list).expect("upload root")
        });
    }
}
