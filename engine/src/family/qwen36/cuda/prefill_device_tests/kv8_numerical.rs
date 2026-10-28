//! Same-entry numerical pin over the stored legacy Int8G32 payload, including its gate.
use super::*;
use crate::family::AttentionKind;
use crate::family::qwen36::cuda::{
    mixer_geometry::AttnGeometry, prefill_plan::ChunkSlot, state_layout::Plane,
};
use crate::kernels::attention_reference::oracle::sdpa_f64;

struct Snapshot {
    p0: usize,
    m: usize,
    geometry: AttnGeometry,
    tokens: usize,
    q: Vec<f32>,
    proj: Vec<f32>,
    out: Vec<f32>,
    kv: [Vec<u8>; 4],
}

fn capture(
    rig: &Rig,
    state: &ResidentState,
    at: &RigChunk<'_>,
    ordinal: usize,
) -> Result<Snapshot, EngError> {
    let read = |range: DeviceRange<'_>| -> Result<Vec<u8>, EngError> {
        let mut bytes = vec![0; range.bytes()];
        range
            .buffer()
            .read_at(&rig.stream, range.offset(), &mut bytes)?;
        Ok(bytes)
    };
    Ok(Snapshot {
        p0: at.p0,
        m: at.m,
        geometry: rig.geometry.attn,
        tokens: state.tokens(),
        q: floats(&read(slot_view(at.arena, at.tensors, ChunkSlot::Q)?)?),
        proj: floats(&read(slot_view(at.arena, at.tensors, ChunkSlot::ProjA)?)?),
        out: floats(&read(slot_view(at.arena, at.tensors, ChunkSlot::AttnOut)?)?),
        kv: [
            read(state.kv_plane(ordinal, Plane::K)?)?,
            read(state.kv_plane(ordinal, Plane::KScales)?)?,
            read(state.kv_plane(ordinal, Plane::V)?)?,
            read(state.kv_plane(ordinal, Plane::VScales)?)?,
        ],
    })
}

fn device_phase() -> Result<Vec<Snapshot>, EngError> {
    let rig = build_rig()?;
    let mut last_full = None;
    for layer in 0..RIG_LAYERS {
        let slot = rig.table.slot(layer)?;
        if slot.kind == AttentionKind::Full {
            last_full = Some((layer + 1, slot.ordinal));
        }
    }
    let (layers, ordinal) = last_full.ok_or_else(|| {
        EngError::BadRequest("Q8 numerical rig has no full-attention layer".to_owned())
    })?;
    let mut state = rig_state(&rig, KvMode::Int8G32)?;
    state.reset(&rig.stream)?;
    let nsm = rig.ctx.mmq_device_facts()?.nsm;
    let (tensors, arena, part) = rig_session(&rig, 128, nsm, CPF_ATTN_PART_CAP)?;
    let bufs = rig_bufs(&rig, &tensors, &arena, &part, 128)?;
    let mixer = rig_mixer(&rig, false)?;
    let moe = rig_moe_slice(&rig, false)?;
    let weights = ChunkWeights::new(&mixer, &moe, &rig.embed);
    // Stop at the last full layer: later layers must not replace its observable slots.
    let model = ChunkModel::leading(&rig.table, &rig.geometry, &rig.dims, weights, layers)?;
    let mut snapshots = Vec::new();
    for (p0, m, seed) in [
        (0, 17, 0x4B56_3846_3634_0001),
        (17, 7, 0x4B56_3846_3634_0002),
    ] {
        for slot in [ChunkSlot::Q, ChunkSlot::ProjA, ChunkSlot::AttnOut] {
            let view = slot_view(&arena, &tensors, slot)?;
            ff_fill(&rig.stream, &view, view.bytes())?;
        }
        open_chunk(&mut state, &rig.stream)?;
        let at = RigChunk {
            bufs: &bufs,
            arena: &arena,
            tensors: &tensors,
            p0,
            m,
            m_sess: 128,
            seed,
        };
        rig_body(&rig, &state, &model, &at)?;
        let status = state.read_prefill_status(&rig.stream)?;
        state.revalidate(&rig.stream, p0 + m, status)?;
        snapshots.push(capture(&rig, &state, &at, ordinal)?);
    }
    Ok(snapshots)
}

fn decoded(codes: &[u8], scales: &[u8], a: &AttnGeometry, keys: usize) -> Vec<f64> {
    // Int8G32 stores head-major signed codes and independent IEEE F16 group scales.
    let groups = a.head_dim / 32;
    assert!(codes.len() >= a.heads_kv * RIG_CAP * a.head_dim);
    assert!(scales.len() >= a.heads_kv * RIG_CAP * groups * 2);
    let mut values = Vec::with_capacity(keys * a.heads_kv * a.head_dim);
    for row in 0..keys {
        for head in 0..a.heads_kv {
            for dim in 0..a.head_dim {
                let cell = head * RIG_CAP + row;
                let code = codes[cell * a.head_dim + dim] as i8;
                let at = (cell * groups + dim / 32) * 2;
                let scale = f16(u16::from_le_bytes([scales[at], scales[at + 1]]));
                assert_ne!(code, i8::MIN, "unwritten/reserved legacy code");
                assert!(scale.is_finite() && scale >= 0.0, "invalid stored scale");
                values.push(code as f64 * scale as f64);
            }
        }
    }
    values
}

fn two_bf16_ulps(reference: f64) -> f64 {
    let exponent = ((reference.abs().to_bits() >> 52) & 0x7ff) as i32 - 1023;
    // BF16 subnormals use 2^-133 spacing; do not clamp smaller references to one.
    2.0 * 2f64.powi((exponent - 7).max(-133))
}

#[test]
fn bf16_bound_tracks_the_reference_binade_without_a_unit_floor() {
    for (reference, bound) in [
        (0.0, 2f64.powi(-132)),
        (2f64.powi(-126), 2f64.powi(-132)),
        (2f64.powi(-20), 2f64.powi(-26)),
        (0.5, 1.0 / 128.0),
        (1.0, 1.0 / 64.0),
        (2.0, 1.0 / 32.0),
    ] {
        assert_eq!(two_bf16_ulps(reference), bound);
        assert_eq!(two_bf16_ulps(-reference), bound);
    }
}

fn gated_reference(snap: &Snapshot) -> Vec<f64> {
    let a = snap.geometry;
    let q: Vec<f64> = snap.q[..snap.m * a.q_width]
        .iter()
        .map(|&x| x as f64)
        .collect();
    let keys = snap.p0 + snap.m;
    let k = decoded(&snap.kv[0], &snap.kv[1], &a, keys);
    let v = decoded(&snap.kv[2], &snap.kv[3], &a, keys);
    let mut reference = sdpa_f64(
        &q,
        &k,
        &v,
        snap.m,
        a.heads_q,
        keys,
        a.head_dim,
        a.heads_q / a.heads_kv,
        a.scale as f64,
        snap.p0 as u32,
        0,
        None,
        None,
    );
    for (index, want) in reference.iter_mut().enumerate() {
        let row = index / a.q_width;
        let head = index % a.q_width / a.head_dim;
        let dim = index % a.head_dim;
        let gate = snap.proj[row * a.ld + head * 2 * a.head_dim + a.head_dim + dim] as f64;
        *want /= 1.0 + (-gate).exp();
    }
    reference
}

#[test]
fn chunk_kv8_attention_matches_f64_at_two_bf16_ulps() {
    if !device_present("chunk_kv8_attention_matches_f64_at_two_bf16_ulps") {
        return;
    }
    let snapshots = device_phase().expect("same-entry Result-only device phase");
    assert_eq!(snapshots.len(), 2);
    for snap in snapshots {
        let a = snap.geometry;
        let values = snap.m * a.q_width;
        assert_eq!(snap.tokens, snap.p0 + snap.m);
        assert!(snap.q.len() >= values && snap.out.len() >= values);
        assert!(snap.proj.len() >= snap.m * a.ld);
        assert_finite("actual Q", &snap.q[..values]);
        assert_finite("actual gate projection", &snap.proj[..snap.m * a.ld]);
        assert_finite("actual AttnOut", &snap.out[..values]);
        let reference = gated_reference(&snap);
        assert_eq!(reference.len(), values);
        assert!(
            reference.iter().all(|value| value.is_finite()),
            "non-finite f64 gated reference"
        );
        let mut worst_ratio = 0.0f64;
        for (vector, row) in reference.chunks_exact(a.head_dim).enumerate() {
            let row_max = row.iter().map(|value| value.abs()).fold(0.0, f64::max);
            let bound = two_bf16_ulps(row_max);
            for (dim, &want) in row.iter().enumerate() {
                let index = vector * a.head_dim + dim;
                let error = (snap.out[index] as f64 - want).abs();
                assert!(
                    error <= bound,
                    "p0={} m={} index={index}: got={} want={want:.17e}, row_max={row_max:.6e}, error={error:.6e}, 2ulp={bound:.6e}",
                    snap.p0,
                    snap.m,
                    snap.out[index],
                );
                worst_ratio = worst_ratio.max(error / bound);
            }
        }
        println!(
            "Q8_SAME_ENTRY_F64 p0={} m={} values={values} worst_2ulp_ratio={worst_ratio:.6e}",
            snap.p0, snap.m
        );
    }
}
