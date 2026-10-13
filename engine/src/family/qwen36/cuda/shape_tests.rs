//! Unit tests of the shape builders of the mixer wrappers (test seam U13). CPU only.
use super::{
    common_tests::dims,
    mixer_geometry::MixerGeometry,
    mixer_shapes::{attn_decode_shape, attn_prep_shape},
    mixer_shapes::{check_delta_eps, conv_shape, delta_shape},
};
use crate::kernels::cuda::resident::attn_prep::SlotMode;

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
