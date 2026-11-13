//! Unit tests of the layer and role table (test seam U13). CPU only.
use super::{
    common_tests::{dims, inventory},
    layer_table::{LayerTable, layer_of},
};
use crate::family::exl3::layer_table::LayerSlot;
use crate::family::{AttentionKind, gguf_model::Role};

#[test]
fn layer_numbers_come_from_the_name_prefix() {
    assert_eq!(layer_of("blk.12.attn_q.weight"), Some(12));
    assert_eq!(layer_of("blk.0.ssm_a"), Some(0));
    assert_eq!(layer_of("output.weight"), None);
    assert_eq!(layer_of("blk.x.attn_q.weight"), None);
    assert_eq!(layer_of("blk.7"), None);
}

#[test]
fn layer_table_names_roles_and_ordinals() {
    let d = dims();
    let table = LayerTable::new(&d, &inventory()).expect("table");
    assert_eq!(table.len(), d.shape.kinds.len());
    let deltas = d.layers_of(AttentionKind::DeltaNet);
    let fulls = d.layers_of(AttentionKind::Full);
    assert_eq!(deltas + fulls, table.len());
    // Ordinals count within the kind and run 0..count without gaps.
    let mut seen = (0, 0);
    for layer in 0..table.len() {
        let LayerSlot { kind, ordinal } = table.slot(layer).expect("slot");
        let counter = if kind == AttentionKind::DeltaNet {
            &mut seen.0
        } else {
            &mut seen.1
        };
        assert_eq!(ordinal, *counter, "layer {layer}");
        *counter += 1;
    }
    assert_eq!(seen, (deltas, fulls));
    let first_full = d
        .shape
        .kinds
        .iter()
        .position(|&k| k == AttentionKind::Full)
        .expect("full");
    let name = table.name(first_full, Role::AttnQ).expect("attn_q");
    assert_eq!(name, format!("blk.{first_full}.attn_q.weight"));
    assert!(
        table.name(first_full, Role::SsmOut).is_err(),
        "a full layer has no ssm_out"
    );
    assert!(
        table.name(0, Role::AttnQ).is_err(),
        "a DeltaNet layer has no attn_q"
    );
    assert_eq!(table.name(0, Role::SsmA).expect("ssm_a"), "blk.0.ssm_a");
    assert!(table.slot(table.len()).is_err());
}

#[test]
fn a_kind_without_a_resident_mixer_is_refused() {
    let mut d = dims();
    d.shape.kinds[1] = AttentionKind::Qsa;
    let error = LayerTable::new(&d, &inventory()).expect_err("Qsa");
    assert!(
        format!("{error:?}").contains("no resident mixer"),
        "{error:?}"
    );
}

#[test]
fn a_single_layer_table_renames_one_layer_and_restarts_its_ordinal() {
    let d = dims();
    let table = LayerTable::new(&d, &inventory()).expect("table");
    let source = d
        .shape
        .kinds
        .iter()
        .position(|&kind| kind == AttentionKind::Full)
        .expect("a full-attention layer");
    let number = table.len();
    let block = table.single(source, number).expect("single");
    assert_eq!(block.len(), 1);
    assert_eq!(
        block.slot(0).expect("slot"),
        LayerSlot {
            kind: AttentionKind::Full,
            ordinal: 0
        }
    );
    let original = table.name(source, Role::AttnQ).expect("name");
    let renamed = block.name(0, Role::AttnQ).expect("name");
    assert_eq!(original, format!("blk.{source}.attn_q.weight"));
    assert_eq!(renamed, format!("blk.{number}.attn_q.weight"));
    for role in [
        Role::AttnK,
        Role::AttnV,
        Role::AttnOutput,
        Role::FfnGateExps,
    ] {
        let name = block.name(0, role).expect("name");
        assert!(name.starts_with(&format!("blk.{number}.")), "{name}");
        assert_eq!(
            name.replace(&format!("blk.{number}."), ""),
            table
                .name(source, role)
                .expect("name")
                .replace(&format!("blk.{source}."), "")
        );
    }
    assert!(
        table.single(table.len(), number).is_err(),
        "a layer outside the table"
    );
}
