//! Header-only inspection of Flash-Next EXL3 packs: a synthetic pack of the stock layout at
//! small widths, its byte total counted independently from the tensor list, and the refusals a
//! changed header must raise. The real pack is checked by the ignored test at the end.

use std::path::Path;

use serde_json::{Value, json};

use super::super::tests::{CONFIG, checkpoint};
use super::{inspect_flash_pack, inspect_flash_pack_listing};
use crate::{
    EngError,
    family::exl3_pack::{Marker, VisitKind},
    weights::exl3::Checkpoint,
};

/// Expert and indexer matrices at 3 bits per weight, the rest at 5, as in the 3.05 pack.
const LOW: u32 = 48;
const HIGH: u32 = 80;

/// The Flash-Next config at the smallest widths the format allows: 4 layers, the last a QSA
/// layer, PLE on layer 1, 3 experts.
fn config() -> Vec<u8> {
    let mut value: Value = serde_json::from_slice(CONFIG).unwrap();
    let text = value["text_config"].as_object_mut().unwrap();
    for (key, number) in [
        ("hidden_size", 128),
        ("hc_count", 2),
        ("hc_lowrank", 128),
        ("vocab_size", 128),
        ("num_hidden_layers", 4),
        ("num_attention_heads", 1),
        ("num_key_value_heads", 1),
        ("head_dim", 128),
        ("indexer_n_heads", 1),
        ("indexer_head_dim", 64),
        ("linear_num_key_heads", 1),
        ("linear_key_head_dim", 128),
        ("linear_num_value_heads", 2),
        ("linear_value_head_dim", 64),
        ("num_experts", 3),
        ("num_experts_per_tok", 2),
        ("moe_intermediate_size", 128),
        ("shared_expert_intermediate_size", 128),
        ("heads_per_ngram", 1),
        ("ple_embed_dim", 128),
    ] {
        text.insert(key.into(), json!(number));
    }
    text.insert(
        "layer_types".into(),
        json!([
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention"
        ]),
    );
    serde_json::to_vec(&value).unwrap()
}

type Entry = (String, &'static str, Vec<u32>);

fn plain(out: &mut Vec<Entry>, name: &str, dtype: &'static str, shape: &[u32]) {
    out.push((name.to_owned(), dtype, shape.to_vec()));
}

fn matrix(out: &mut Vec<Entry>, prefix: &str, (k, n): (u32, u32), words: u32) {
    plain(
        out,
        &format!("{prefix}.trellis"),
        "I16",
        &[k / 16, n / 16, words],
    );
    plain(out, &format!("{prefix}.suh"), "F16", &[k]);
    plain(out, &format!("{prefix}.svh"), "F16", &[n]);
    plain(out, &format!("{prefix}.mul1"), "I32", &[]);
}

fn hyper(out: &mut Vec<Entry>, prefix: &str, inject: bool) {
    plain(out, &format!("{prefix}.hc_norm.weight"), "F16", &[256]);
    plain(
        out,
        &format!("{prefix}.input_mix_weight_down.weight"),
        "F16",
        &[128, 256],
    );
    plain(
        out,
        &format!("{prefix}.input_mix_weight_up.weight"),
        "F16",
        &[256, 128],
    );
    if inject {
        plain(
            out,
            &format!("{prefix}.block_inject_weight.weight"),
            "F16",
            &[2, 256],
        );
    }
}

fn layer(index: usize, suffix: &str) -> String {
    format!("model.language_model.layers.{index}.{suffix}")
}

/// The stock layout of the small config, every tensor in its stored dtype.
fn pack() -> Vec<Entry> {
    let mut out = Vec::new();
    plain(
        &mut out,
        "model.language_model.embed_tokens.weight",
        "BF16",
        &[128, 128],
    );
    matrix(&mut out, "lm_head", (128, 128), HIGH);
    hyper(
        &mut out,
        "model.language_model.hyper_connection_mixer",
        false,
    );
    for index in 0..4 {
        let at = |suffix: &str| layer(index, suffix);
        hyper(&mut out, &at("attn_hyper_connection"), true);
        if index == 3 {
            matrix(&mut out, &at("self_attn.q_proj"), (128, 256), HIGH);
            matrix(&mut out, &at("self_attn.k_proj"), (128, 128), HIGH);
            matrix(&mut out, &at("self_attn.v_proj"), (128, 128), HIGH);
            matrix(&mut out, &at("self_attn.o_proj"), (128, 128), HIGH);
            matrix(
                &mut out,
                &at("self_attn.indexer.index_qk_proj"),
                (128, 128),
                LOW,
            );
            for norm in ["q_norm", "k_norm"] {
                plain(
                    &mut out,
                    &at(&format!("self_attn.{norm}.weight")),
                    "BF16",
                    &[128],
                );
            }
            for norm in ["q_layernorm", "k_layernorm"] {
                plain(
                    &mut out,
                    &at(&format!("self_attn.indexer.{norm}.weight")),
                    "BF16",
                    &[64],
                );
            }
        } else {
            matrix(&mut out, &at("linear_attn.in_proj_qkv"), (128, 384), HIGH);
            matrix(&mut out, &at("linear_attn.in_proj_z"), (128, 128), HIGH);
            matrix(&mut out, &at("linear_attn.out_proj"), (128, 128), HIGH);
            for gate in ["in_proj_a", "in_proj_b"] {
                plain(
                    &mut out,
                    &at(&format!("linear_attn.{gate}.weight")),
                    "F16",
                    &[2, 128],
                );
            }
            plain(&mut out, &at("linear_attn.A_log"), "BF16", &[2]);
            plain(&mut out, &at("linear_attn.dt_bias"), "BF16", &[2]);
            plain(
                &mut out,
                &at("linear_attn.conv1d.weight"),
                "BF16",
                &[384, 1, 4],
            );
            plain(&mut out, &at("linear_attn.norm.weight"), "BF16", &[64]);
        }
        if index == 1 {
            plain(&mut out, &at("ple.key_proj.weight"), "F16", &[256, 128]);
            plain(&mut out, &at("ple.value_proj.weight"), "F16", &[128, 128]);
            for norm in ["norm_conv", "norm_key", "norm_query"] {
                plain(&mut out, &at(&format!("ple.{norm}.weight")), "BF16", &[256]);
            }
            plain(&mut out, &at("ple.conv1d.weight"), "F16", &[256, 1, 4]);
        }
        hyper(&mut out, &at("mlp_hyper_connection"), true);
        plain(&mut out, &at("mlp.gate.weight"), "F16", &[3, 128]);
        plain(
            &mut out,
            &at("mlp.shared_expert_gate.weight"),
            "F16",
            &[1, 128],
        );
        for part in ["gate_proj", "up_proj", "down_proj"] {
            for expert in 0..3 {
                let prefix = at(&format!("mlp.experts.{expert}.{part}"));
                matrix(&mut out, &prefix, (128, 128), LOW);
            }
            matrix(
                &mut out,
                &at(&format!("mlp.shared_expert.{part}")),
                (128, 128),
                HIGH,
            );
        }
    }
    out
}

fn width(dtype: &str) -> u64 {
    match dtype {
        "F16" | "BF16" | "I16" => 2,
        _ => 4,
    }
}

/// The device bytes of one entry by the module's rule, decided from the name alone: trellises
/// and the tensors the kernels take as stored count as stored, `suh`, `svh` and the remaining
/// plain tensors as F32, markers and the host-only PLE norms and convolution not at all.
fn device_bytes(name: &str, dtype: &str, shape: &[u32]) -> u64 {
    let values = shape.iter().map(|&d| u64::from(d)).product::<u64>();
    let kept = ["embed_tokens", "hyper_connection", "_proj.weight"];
    // The host PLE step reads these: no device bytes.
    let host_only = name.contains(".ple.norm_") || name.contains(".ple.conv1d");
    if name.ends_with(".mul1") || host_only {
        0
    } else if name.ends_with(".trellis") || kept.iter().any(|part| name.contains(part)) {
        values * width(dtype)
    } else {
        values * 4
    }
}

fn open(entries: &[Entry]) -> crate::weights::StreamedCheckpoint {
    let tensors = entries
        .iter()
        .map(|(name, dtype, shape)| {
            let bytes =
                shape.iter().map(|&d| d as usize).product::<usize>() * width(dtype) as usize;
            (name.as_str(), *dtype, shape.clone(), vec![0u8; bytes])
        })
        .collect::<Vec<_>>();
    checkpoint(&tensors)
}

fn refusal(entries: &[Entry], config: &[u8]) -> String {
    match inspect_flash_pack(&open(entries), config) {
        Err(EngError::Checkpoint(message)) => message,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_stock_pack_matches_the_config_and_an_independent_byte_sum() {
    let entries = pack();
    let report = inspect_flash_pack(&open(&entries), &config()).unwrap();
    let total = entries
        .iter()
        .map(|(name, dtype, shape)| device_bytes(name, dtype, shape))
        .sum::<u64>();
    let experts = entries
        .iter()
        .filter(|(name, ..)| name.contains(".mlp.experts."))
        .map(|(name, dtype, shape)| device_bytes(name, dtype, shape))
        .sum::<u64>();
    assert_eq!(report.pack.device_bytes, total);
    assert_eq!(report.pack.expert_bytes, experts);
    // lm_head, 3 DeltaNet layers of 3 and a QSA layer of 5, 4 layers of 9 experts and 3 shared.
    assert_eq!(report.pack.matrices, 1 + 3 * 3 + 5 + 4 * 12);
    let low = 1 + 4 * 9;
    assert_eq!(
        report.pack.rates.into_iter().collect::<Vec<_>>(),
        [
            ((LOW / 8) as u8, low),
            ((HIGH / 8) as u8, 1 + 9 + 4 + 4 * 3)
        ]
    );
    assert_eq!(report.pack.codebook, Marker::Mul1);
}

/// The recording walk returns the report of the plain walk, and its list is the pack: every
/// tensor name of the pack is a visited plain tensor or a component of a visited matrix, and the
/// visits' bytes add up to the device bytes.
#[test]
fn the_listing_is_the_pack_and_adds_up_to_the_report() {
    let entries = pack();
    let source = open(&entries);
    let plain_report = inspect_flash_pack(&source, &config()).unwrap();
    let (report, visits) = inspect_flash_pack_listing(&source, &config()).unwrap();
    assert_eq!(report, plain_report);
    assert_eq!(
        visits.iter().map(|v| v.bytes).sum::<u64>(),
        report.pack.device_bytes
    );
    let matrices = visits
        .iter()
        .filter(|v| matches!(v.kind, VisitKind::Matrix { .. }))
        .count();
    assert_eq!(matrices, report.pack.matrices);
    let visited: std::collections::BTreeSet<&str> =
        visits.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(visited.len(), visits.len(), "a tensor visited twice");
    for (name, ..) in &entries {
        let owner = name
            .rsplit_once('.')
            .filter(|(_, suffix)| ["trellis", "suh", "svh", "mul1"].contains(suffix))
            .map_or(name.as_str(), |(prefix, _)| prefix);
        assert!(visited.contains(owner), "{name} is not in the listing");
    }
    // The gates are F32 on the device now.
    let gate = visits
        .iter()
        .find(|v| v.name.ends_with("in_proj_a.weight"))
        .expect("in_proj_a");
    assert!(matches!(gate.kind, VisitKind::Plain { keep: false, .. }));
}

/// Each case changes one fact of the stock pack; the refusal names it.
#[test]
fn a_pack_that_differs_from_the_config_is_refused_by_name() {
    let mixer = "model.language_model.hyper_connection_mixer";
    let rename = |from: &str, to: &str| -> Vec<Entry> {
        pack()
            .into_iter()
            .map(|(name, dtype, shape)| (name.replace(from, to), dtype, shape))
            .collect()
    };
    let retype = |target: String, dtype: &'static str| -> Vec<Entry> {
        pack()
            .into_iter()
            .map(|(name, old, shape)| {
                let dtype = if name == target { dtype } else { old };
                (name, dtype, shape)
            })
            .collect()
    };
    let without = |target: String| -> Vec<Entry> {
        pack()
            .into_iter()
            .filter(|(name, ..)| *name != target)
            .collect()
    };
    let reshape = |target: String, shape: Vec<u32>| -> Vec<Entry> {
        pack()
            .into_iter()
            .map(|(name, dtype, old)| {
                let shape = if name == target { shape.clone() } else { old };
                (name, dtype, shape)
            })
            .collect()
    };
    let inject = layer(2, "mlp_hyper_connection.block_inject_weight.weight");
    let mut qsa_first: Value = serde_json::from_slice(&config()).unwrap();
    qsa_first["text_config"]["layer_types"][0] = json!("full_attention");
    let qsa_first = serde_json::to_vec(&qsa_first).unwrap();
    let cases: Vec<(&str, Vec<Entry>, Vec<u8>, String)> = vec![
        (
            "no mixer",
            without(format!("{mixer}.hc_norm.weight")),
            config(),
            format!("missing tensor {mixer}.hc_norm.weight"),
        ),
        (
            "a site of two dtypes",
            retype(inject.clone(), "BF16"),
            config(),
            format!("{inject} is Bf16, the rest of"),
        ),
        (
            "an F32 site",
            retype(format!("{mixer}.hc_norm.weight"), "F32"),
            config(),
            "the kernel needs F16 or Bf16".into(),
        ),
        (
            "an inject of another branch count",
            reshape(inject.clone(), vec![4, 256]),
            config(),
            format!("{inject} is [4, 256], the model needs [2, 256]"),
        ),
        (
            "an indexer without its key head",
            reshape(
                layer(3, "self_attn.indexer.index_qk_proj.trellis"),
                vec![8, 4, LOW],
            ),
            config(),
            "the model needs I16 [8, 8, w]".into(),
        ),
        (
            "no PLE block on the PLE layer",
            without(layer(1, "ple.key_proj.weight")),
            config(),
            format!("missing tensor {}", layer(1, "ple.key_proj.weight")),
        ),
        (
            "a DeltaNet layer the config calls QSA",
            pack(),
            qsa_first,
            format!("missing tensor {}", layer(0, "self_attn.q_proj.trellis")),
        ),
        (
            "an expert of another rate",
            reshape(layer(2, "mlp.experts.1.up_proj.trellis"), vec![8, 8, HIGH]),
            config(),
            "layer 2 Up experts differ in rate".into(),
        ),
        (
            "another codebook",
            rename("self_attn.o_proj.mul1", "self_attn.o_proj.mcg"),
            config(),
            format!(
                "{} has another codebook than the pack",
                layer(3, "self_attn.o_proj")
            ),
        ),
        (
            "the MLX layout",
            rename("model.language_model.", "language_model.model."),
            config(),
            "missing tensor model.language_model.embed_tokens.weight".into(),
        ),
    ];
    for (what, entries, config, wanted) in cases {
        let message = refusal(&entries, &config);
        assert!(message.contains(&wanted), "{what}: {message}");
    }
}

/// The real pack: `FIFI_FLASH_NEXT_EXL3_PACK` names a pack directory (the shard files may be
/// sparse copies that hold only their headers), `FIFI_FLASH_NEXT_EXL3_EXPECT_BYTES` and
/// `FIFI_FLASH_NEXT_EXL3_EXPECT_EXPERT_BYTES` the device bytes an independent sum over its
/// headers gives by the module's rule. Prints the report.
#[test]
#[ignore = "reads the headers of a real pack in FIFI_FLASH_NEXT_EXL3_PACK"]
fn the_real_pack_matches_the_config_and_its_independent_byte_sum() {
    let var = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("{name}"));
    let checkpoint = Checkpoint::open(Path::new(&var("FIFI_FLASH_NEXT_EXL3_PACK"))).expect("open");
    let report = inspect_flash_pack(&checkpoint.source, &checkpoint.model_config).expect("fits");
    println!("REPORTED: {report:?}");
    let expected = |name: &str| var(name).parse::<u64>().unwrap();
    assert_eq!(
        report.pack.device_bytes,
        expected("FIFI_FLASH_NEXT_EXL3_EXPECT_BYTES")
    );
    assert_eq!(
        report.pack.expert_bytes,
        expected("FIFI_FLASH_NEXT_EXL3_EXPECT_EXPERT_BYTES")
    );
    assert_eq!(report.pack.codebook, Marker::Mul1);
}
