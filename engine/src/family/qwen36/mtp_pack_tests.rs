//! Header-only inspection of the MTP head: a synthetic pack at small widths with the head's
//! tensors added, the refusals a changed header must raise, and an ignored check of a real pack.

use std::path::Path;

use super::{
    exl3_pack::{ExpertPart, PackReport},
    fixtures::{self, Entry},
    host::{AttnDims, DeltaDims, HostShape, MoeDims},
    mtp_pack::{FC_PREFIX, NORM_NAME, PRE_FC_NORM_EMBEDDING, PRE_FC_NORM_HIDDEN, inspect_mtp_pack},
};
use crate::family::exl3_pack::Marker;
use crate::{
    EngError,
    family::{AttentionKind, QWEN3_6_35B_A3B},
    weights::exl3::Checkpoint,
};

const HIDDEN: u32 = 128;
const INTER: u32 = 128;
const EXPERTS: usize = 3;
const WORDS: u32 = 64;

fn shape() -> HostShape {
    HostShape {
        hidden: HIDDEN as usize,
        vocab: 128,
        kinds: vec![AttentionKind::Full],
        eps: 1e-6,
        delta: DeltaDims {
            key_heads: 2,
            key_dim: 128,
            value_heads: 4,
            value_dim: 128,
            conv_kernel: 4,
        },
        attn: AttnDims {
            heads_q: 2,
            heads_kv: 1,
            head_dim: 128,
            rotary_dim: 64,
            theta: 1e7,
        },
        dense_inter: None,
        moe: MoeDims {
            experts: EXPERTS,
            top_k: 2,
            inter: INTER as usize,
            shared_inter: INTER as usize,
            renorm: true,
        },
    }
}

fn entry(name: &str, dtype: &'static str, shape: &[u32]) -> Entry {
    let width = if dtype == "F16" || dtype == "BF16" || dtype == "I16" {
        2
    } else {
        4
    };
    let bytes = shape.iter().map(|&d| d as usize).product::<usize>().max(1) * width;
    (name.to_owned(), dtype, shape.to_vec(), vec![0; bytes])
}

fn matrix(out: &mut Vec<Entry>, prefix: &str, (k, n): (u32, u32)) {
    out.push(entry(
        &format!("{prefix}.trellis"),
        "I16",
        &[k / 16, n / 16, WORDS],
    ));
    out.push(entry(&format!("{prefix}.suh"), "F16", &[k]));
    out.push(entry(&format!("{prefix}.svh"), "F16", &[n]));
    out.push(entry(&format!("{prefix}.mul1"), "I32", &[]));
}

/// The tensors of a one-block head.
fn head() -> Vec<Entry> {
    let h = HIDDEN;
    let at = |suffix: &str| format!("mtp.layers.0.{suffix}");
    let mut out = vec![
        entry(PRE_FC_NORM_HIDDEN, "BF16", &[h]),
        entry(PRE_FC_NORM_EMBEDDING, "BF16", &[h]),
        entry(NORM_NAME, "BF16", &[h]),
        entry(&at("input_layernorm.weight"), "BF16", &[h]),
        entry(&at("post_attention_layernorm.weight"), "BF16", &[h]),
        entry(&at("mlp.gate.weight"), "BF16", &[EXPERTS as u32, h]),
        entry(&at("mlp.shared_expert_gate.weight"), "BF16", &[1, h]),
    ];
    matrix(&mut out, FC_PREFIX, (2 * h, h));
    let (q, kv) = (2 * 128, 128);
    matrix(&mut out, &at("self_attn.q_proj"), (h, 2 * q));
    matrix(&mut out, &at("self_attn.k_proj"), (h, kv));
    matrix(&mut out, &at("self_attn.v_proj"), (h, kv));
    matrix(&mut out, &at("self_attn.o_proj"), (q, h));
    for norm in ["q_norm", "k_norm"] {
        out.push(entry(
            &at(&format!("self_attn.{norm}.weight")),
            "BF16",
            &[128],
        ));
    }
    for (part, name) in [
        (ExpertPart::Gate, "gate_proj"),
        (ExpertPart::Up, "up_proj"),
        (ExpertPart::Down, "down_proj"),
    ] {
        let dims = match part {
            ExpertPart::Down => (INTER, h),
            _ => (h, INTER),
        };
        for e in 0..EXPERTS {
            matrix(&mut out, &at(&format!("mlp.experts.{e}.{name}")), dims);
        }
        matrix(&mut out, &at(&format!("mlp.shared_expert.{name}")), dims);
    }
    out
}

fn walk(entries: Vec<Entry>) -> Result<PackReport, EngError> {
    inspect_mtp_pack(&fixtures::checkpoint(entries), &shape(), 1)
}

fn position(entries: &[Entry], name: &str) -> usize {
    entries
        .iter()
        .position(|e| e.0 == name)
        .unwrap_or_else(|| panic!("{name}"))
}

#[test]
fn a_head_of_the_model_shape_is_counted_from_its_headers() {
    let entries = head();
    // Trellises as stored, markers nothing, every other half-width tensor widened to F32.
    let expected: u64 = entries
        .iter()
        .map(|(name, _, _, bytes)| match name {
            n if n.ends_with(".mul1") => 0,
            n if n.ends_with(".trellis") => bytes.len() as u64,
            _ => bytes.len() as u64 * 2,
        })
        .sum();
    let report = walk(entries).expect("the head fits the shape");
    assert_eq!(report.device_bytes, expected);
    assert_eq!(report.codebook, Marker::Mul1);
    // fc + 4 attention matrices + 3 projections x (3 experts + 1 shared).
    assert_eq!(report.matrices, 1 + 4 + 3 * (EXPERTS + 1));
}

#[test]
fn a_pack_without_the_head_is_refused_by_name() {
    let message = match walk(vec![entry("lm_head.suh", "F16", &[HIDDEN])]) {
        Err(EngError::Checkpoint(message)) => message,
        other => panic!("expected a refusal, got {other:?}"),
    };
    assert!(
        message.contains("mtp.pre_fc_norm_hidden.weight"),
        "{message}"
    );
}

#[test]
fn each_header_change_is_a_named_refusal() {
    let base = head();
    let mut cases: Vec<(&str, Vec<Entry>, &str)> = Vec::new();

    let mut e = base.clone();
    let at = position(&e, "mtp.fc.trellis");
    e[at] = entry("mtp.fc.trellis", "I16", &[8, 8, WORDS]);
    cases.push(("an fc of half the input width", e, "the model needs"));

    let mut e = base.clone();
    e.remove(position(&e, "mtp.layers.0.mlp.experts.1.up_proj.trellis"));
    cases.push(("a missing expert", e, "missing tensor"));

    let mut e = base.clone();
    let at = position(&e, "mtp.layers.0.mlp.experts.2.gate_proj.mul1");
    e[at].0 = "mtp.layers.0.mlp.experts.2.gate_proj.mcg".into();
    cases.push(("a second codebook", e, "another codebook"));

    let mut e = base.clone();
    let at = position(&e, NORM_NAME);
    e[at] = entry(NORM_NAME, "BF16", &[64]);
    cases.push(("a short output norm", e, "the model needs"));

    for (what, entries, wanted) in cases {
        let message = match walk(entries) {
            Err(EngError::Checkpoint(message)) => message,
            other => panic!("{what}: expected a refusal, got {other:?}"),
        };
        assert!(message.contains(wanted), "{what}: {message}");
    }
}

#[test]
fn a_head_with_no_block_is_refused() {
    let source = fixtures::checkpoint(head());
    assert!(inspect_mtp_pack(&source, &shape(), 0).is_err());
}

#[test]
fn a_second_block_needs_its_own_tensors() {
    let source = fixtures::checkpoint(head());
    let message = match inspect_mtp_pack(&source, &shape(), 2) {
        Err(EngError::Checkpoint(message)) => message,
        other => panic!("expected a refusal, got {other:?}"),
    };
    assert!(message.contains("mtp.layers.1."), "{message}");
}

/// The real pack: `FIFI_MTP_CHECKPOINT` names the directory that holds the `mtp.*` tensors.
/// Prints the report.
#[test]
#[ignore = "reads the headers of a real pack in FIFI_MTP_CHECKPOINT"]
fn the_real_head_matches_the_model_shape() {
    let dir = std::env::var_os("FIFI_MTP_CHECKPOINT").expect("FIFI_MTP_CHECKPOINT");
    let checkpoint = Checkpoint::open(Path::new(&dir)).expect("open the pack");
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).unwrap();
    let report = inspect_mtp_pack(&checkpoint.source, &shape, 1).expect("the head fits the model");
    println!("{report:?}");
    // fc + q, k, v, o + 3 projections x (256 experts + 1 shared).
    assert_eq!(report.matrices, 1 + 4 + 3 * 257);
    assert_eq!(report.codebook, Marker::Mul1);
}
