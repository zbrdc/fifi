//! Unit tests of the layer and role table (test seam U13). CPU only.
use super::{
    common_tests::{dims, inventory},
    layer_table::{LayerSlot, LayerTable, layer_of},
};
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
