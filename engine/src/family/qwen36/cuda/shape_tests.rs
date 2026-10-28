//! Unit tests of the shape builders of the mixer wrappers (test seam U13). CPU only.
use super::{
    common_tests::dims,
    mixer_geometry::MixerGeometry,
    mixer_shapes::{attn_decode_shape, attn_prep_shape},
    mixer_shapes::{attn_q8_shape, kv8_shape},
    mixer_shapes::{check_delta_eps, conv_shape, delta_shape},
};
use crate::kernels::cuda::resident::{attn_prep::SlotMode, kv8_append::Kv8CodeScale};

fn geometry() -> MixerGeometry {
    MixerGeometry::new(&dims()).expect("geometry")
}

#[test]
fn the_conv_and_delta_shapes_pin_every_scalar_argument() {
    let g = geometry();
    let c = conv_shape(&g);
    assert_eq!((c.m, c.channels, c.ld), (1, 8192, 12352));
    let d = delta_shape(&g);
    assert_eq!((d.m, d.n_k_heads, d.n_v_heads), (1, 16, 32));
    assert_eq!((d.conv_ld, d.ld), (8192, 12352));
    assert_eq!((d.z_off, d.beta_off, d.alpha_off), (8192, 12288, 12320));
}

#[test]
fn the_delta_kernels_take_one_epsilon() {
    let one = 1.0e-6_f32;
    check_delta_eps(one).expect("the GGUF value");
    for bad in [
        1.0e-5,
        f32::NAN,
        f32::from_bits(one.to_bits() + 1),
        f32::from_bits(one.to_bits() - 1),
    ] {
        let error = check_delta_eps(bad).expect_err("another epsilon");
        assert!(format!("{error:?}").contains("epsilon"), "{error:?}");
    }
}

#[test]
fn the_attention_shapes_pin_every_scalar_argument() {
    let g = geometry();
    let p = attn_prep_shape(&g, 1024);
    assert_eq!((p.m, p.n_head, p.n_kv_head, p.head_dim), (1, 16, 2, 256));
    assert_eq!((p.n_rot, p.has_gate, p.rows, p.ld), (64, true, 1024, 9216));
    assert!(matches!(p.slot, SlotMode::Position));
    let a = attn_decode_shape(&g, 1024);
    assert_eq!((a.m, a.n_head, a.n_kv_head, a.head_dim), (1, 16, 2, 256));
    assert_eq!((a.cache_rows, a.ld), (1024, 9216));
    assert_eq!(g.attn.scale, 0.0625);
    assert_eq!(g.attn.rope_base, 1.0e7);
}

#[test]
fn the_attention_shapes_forward_the_kv_capacity_they_are_given() {
    // Packet RD-9b3b section 2 pins `rows == KV capacity` and `cache_rows == capacity`: the
    // builders must forward the capacity the executor passes (`layout().kv.capacity`), not a
    // constant. The pin uses a capacity unlike the plan fixture's 1024 (`context_tokens`), so a
    // hard-coded 1024 cannot pass it.
    let capacity = 1536;
    let p = attn_prep_shape(&geometry(), capacity);
    assert_eq!(p.rows, capacity, "prep must forward the KV capacity");
    let a = attn_decode_shape(&geometry(), capacity);
    assert_eq!(
        a.cache_rows, capacity,
        "decode must forward the KV capacity"
    );
}

/// The Int8G32 attention shapes (KV8-B B3, pinned per the B2/B3 review item 2): every scalar
/// argument of the append and the q8 decode, the capacity forwarding, and the equality the
/// step's status argument depends on: the append bounds the pos word by the SAME cell count
/// the rows entry of `attn_prep` bounds it by (`kv8_shape.cells == attn_prep_shape.rows`).
#[test]
fn the_int8_attention_shapes_pin_every_scalar_argument() {
    let g = geometry();
    let k = kv8_shape(&g, 1024);
    assert_eq!(k.code_scale, Kv8CodeScale::StoredHalf);
    assert_eq!(
        (k.rows, k.n_kv_head, k.head_dim, k.cells),
        (1, 2, 256, 1024)
    );
    let q = attn_q8_shape(&g, 1024);
    assert_eq!(
        (q.m, q.n_head, q.n_kv_head, q.head_dim, q.cells, q.ld),
        (1, 16, 2, 256, 1024, 9216)
    );
}

#[test]
fn the_int8_shapes_forward_the_kv_capacity_and_match_the_prep_rows() {
    // Like the Fp16 pin above: a capacity unlike the 1024 of the scalar pin, so a hard-coded
    // cell count or a wrong field (head_dim, rows) cannot pass it.
    let capacity = 1536;
    let k = kv8_shape(&geometry(), capacity);
    assert_eq!(k.code_scale, Kv8CodeScale::StoredHalf);
    let q = attn_q8_shape(&geometry(), capacity);
    let p = attn_prep_shape(&geometry(), capacity);
    assert_eq!(k.cells, capacity, "the append must forward the cell count");
    assert_eq!(q.cells, capacity, "the decode must forward the cell count");
    // The status argument of the append shares the prep's guard: the two bounds are one
    // number, the KV capacity, never two constants.
    assert_eq!(
        k.cells, p.rows,
        "the cell bound of the append equals the row bound of the rows entry"
    );
}
