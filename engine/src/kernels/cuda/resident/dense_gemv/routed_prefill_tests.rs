use super::*;
use crate::kernels::cuda::resident::dense_gemv::routed_prefill::{
    DeviceRoutedPrefill, routed_prefill_scratch_bytes,
};
use crate::weights::{GgufType, dequantize_into};

/// Distinct routing/scales, two tiles of the hot expert, empty experts, and a tail token.
/// One active K coordinate per token; the down rows read a constant activated row.
#[test]
fn device_routed_stage_matches_scalar_projection_and_stable_sort() {
    if !testkit::device_present("device_routed_stage_matches_scalar_projection_and_stable_sort") {
        return;
    }
    let rig = Rig::new();
    let ctx = &rig.ctx;
    let rows = 17;
    let slots = rows * 8;
    let mut gb = [0u8; 66];
    gb[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    let mut db = [0u8; 18];
    db[..2].copy_from_slice(&0x1400u16.to_le_bytes());
    let gate = ctx.upload(&gb.repeat(128 * 2304 * 9)).unwrap();
    let down = ctx.upload(&db.repeat(128 * 2304 * 18)).unwrap();
    let mut gv = [0.; 256];
    let mut dv = [0.; 64];
    dequantize_into(GgufType::Iq2Xxs, &gb, &mut gv).unwrap();
    dequantize_into(GgufType::Q20, &db, &mut dv).unwrap();
    let ids = (0..slots)
        .map(|s| {
            if s % 8 == 0 {
                0i32
            } else {
                1 + ((s * 13) % 127) as i32
            }
        })
        .collect::<Vec<_>>();
    let gs = (0..128).map(|e| 0.25 + e as f32 / 128.).collect::<Vec<_>>();
    let ds = (0..128).map(|e| 1. + e as f32 / 256.).collect::<Vec<_>>();
    let arrays = [gs.as_slice(), ds.as_slice()].map(|v| ctx.upload(&f32_bytes(v)).unwrap());
    let mut input = vec![0.; rows * 2304];
    for (r, row) in input.chunks_mut(2304).enumerate() {
        row[0] = 0.25 + r as f32 / 32.;
    }
    let picks = ctx
        .upload(
            &ids.iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let weights = ctx.upload(&f32_bytes(&vec![0.125; slots])).unwrap();
    let x = ctx.upload(&f32_bytes(&input)).unwrap();
    let scratch = Padded::new(
        ctx,
        &sentinel(routed_prefill_scratch_bytes(rows).unwrap()),
        0,
    );
    let out = Padded::new(ctx, &sentinel(input.len() * 4), 0);
    let status = Padded::new(ctx, &[0; 4], 0);
    fn all(b: &DeviceBuffer) -> DeviceRange<'_> {
        b.range(0, b.bytes()).unwrap()
    }
    let r = DeviceRoutedPrefill {
        rows,
        x: all(&x),
        ids: all(&picks),
        weights: all(&weights),
        out: out.all(),
        scratch: scratch.all(),
        status: status.all(),
    };
    ctx.routed_prefill(
        &rig.stream,
        r,
        [all(&gate), all(&down)],
        arrays.each_ref().map(all),
        [16, 42],
        [1., 1.],
    )
    .unwrap();
    rig.stream.synchronize().unwrap();
    assert_eq!(status.read(ctx, "status"), [0; 4]);
    let bytes = scratch.read(ctx, "scratch");
    let word = |i: usize| i32::from_le_bytes(bytes[i * 4..][..4].try_into().unwrap());
    let mut ordered = (0..slots).collect::<Vec<_>>();
    ordered.sort_by_key(|&s| ids[s]);
    for (pos, s) in ordered.into_iter().enumerate() {
        assert_eq!(word(129 + pos), s as i32);
        assert_eq!(word(129 + slots + s), pos as i32);
    }
    let result = testkit::floats(&out.read(ctx, "routed out"));
    for (r, row) in result.as_chunks::<2304>().0.iter().enumerate() {
        let mut expected = 0.;
        for &e in &ids[r * 8..r * 8 + 8] {
            let e = e as usize;
            let g = input[r * 2304] * gv[0] * gs[e];
            let act = g.tanh().powi(2) / (1. + (-g).exp());
            let mid = half::f16::from_f32(act).to_f32();
            expected += 0.125 * ds[e] * mid * (18. * dv.iter().sum::<f32>());
        }
        assert!(
            row.iter().all(|&v| (v - expected).abs() < 0.001),
            "row {r} expected {expected} got {}",
            row[0]
        );
    }
}
