//! Tiny native cache proof. The parent must allocate hardware before running it.
use super::*;
use crate::{
    kernels::MlxKernels,
    tensor::{Dtype, Tensor},
};

#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn bf16_cache_window_and_reset_preserve_chronological_rows() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let mut state = State::new();
    for pos in 0..=514 {
        let values: Vec<f32> = (0..4)
            .flat_map(|head| vec![(pos % 64 + head) as f32; 128])
            .collect();
        let host = Tensor::try_from_host(
            Dtype::F32,
            vec![4, 1, 128],
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
        let row = k.import_tensor(&host).unwrap();
        let plan = CacheStep::new(2, pos).unwrap();
        let (keys, values) =
            Plane::append(&mut k, &mut state.planes[2], &plan, (&row, &row)).unwrap();
        if [0, 512, 513, 514].contains(&pos) {
            let start = pos.saturating_sub(512);
            let count = (pos + 1 - start) as usize;
            assert_eq!(keys.shape(), [4, count as u32, 128]);
            for tensor in [&keys, &values] {
                let tensor = k.cast(tensor, Dtype::F32).unwrap();
                let host = k.readback(&tensor).unwrap();
                let actual: Vec<_> = host
                    .host_bytes()
                    .unwrap()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                for head in 0..4 {
                    for (row, absolute) in (start..=pos).enumerate() {
                        assert!(
                            actual[(head * count + row) * 128..(head * count + row + 1) * 128]
                                .iter()
                                .all(|&v| v == (absolute % 64 + head as u32) as f32)
                        );
                    }
                }
            }
        }
        state.pos = pos + 1;
    }
    state.valid = false;
    state.reset();
    assert_eq!(state.pos, 0);
    assert!(state.valid);
    assert!(state.planes.iter().all(Option::is_none));
}

/// F3: the ring-view prefix slice for vendor decode. Full-layer one-row
/// chunks hand attention [4, len, 128] while FIFI_FLINT_MLX_DECODE_ATTN=1
/// arms, the whole capacity plane once the flag is off, SWA layers keep
/// the bounded union, and multi-row chunks never meet the ring arm.
/// Checkpoints pin the growth boundary (4096 -> 8192) with hard-coded
/// shapes; pos 4096 also proves the slice end includes the row just
/// appended by reading its bytes back.
#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn ring_view_vendor_slice_tracks_growth_gate_and_layers() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    // BF16-exact markers (<= 130) that still separate pos 4096 from pos 0:
    // the high bit flips once per 4096 rows, the low 6 bits cycle per 64.
    let marker = |pos: u32, head: u32| (((pos / 4096) % 2) * 64 + pos % 64 + head) as f32;

    let row_for = |k: &mut MlxKernels, pos: u32| {
        let values: Vec<f32> = (0..4)
            .flat_map(|head| vec![marker(pos, head); 128])
            .collect();
        let host = Tensor::try_from_host(
            Dtype::F32,
            vec![4, 1, 128],
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
        k.import_tensor(&host).unwrap()
    };

    // Full layer 0: shapes at each checkpoint. pos 0 slices to len 1; 4095
    // fills the first capacity exactly; 4096 crosses the growth boundary
    // (capacity 8192, slice 4097); 8191 is exactly full again (the slice
    // would be the plane, so the share shows); with the flag OFF the plane
    // is whole at every depth - checked at 8192 against the grown 16384.
    let armed = crate::kernels::mlx::decode_attn_flagged();
    let mut state = State::new();
    for pos in 0u32..=8192 {
        state.ensure_full_capacity(&mut k, pos + 1).unwrap();
        let plan = ChunkCachePlan::new(0, pos, 1, state.full_capacity).unwrap();
        let row = row_for(&mut k, pos);
        let (keys, values, guard) =
            Plane::prepare_chunk(&mut k, &mut state.planes[0], &plan, &row, &row, true).unwrap();
        if armed {
            if let Some((_, want)) = [
                (0u32, 1u32),
                (4095, 4096),
                (4096, 4097),
                (8191, 8192),
                (8192, 8193),
            ]
            .iter()
            .find(|(at, _)| *at == pos)
            {
                assert_eq!(keys.shape(), [4, *want, 128], "sliced at pos {pos}");
            }
        } else if pos == 8192 {
            assert_eq!(keys.shape(), [4, 16384, 128], "flag off shares the plane");
        }
        if armed && pos == 4096 {
            // The slice end includes the appended row: last row is this
            // step's marker, row 0 the first, and heads stay apart.
            let keys_f32 = k.cast(&keys, Dtype::F32).unwrap();
            let host = k.readback(&keys_f32).unwrap();
            let actual: Vec<f32> = host
                .host_bytes()
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            for head in 0..4u32 {
                // Last row of head h sits at row offset 4096 of the slice;
                // the high-bit marker keeps 4096 distinct from 0 so this
                // really is the appended row, not row 0 again.
                let base = (head as usize * 4097 + 4096) * 128;
                assert!(
                    actual[base..base + 128]
                        .iter()
                        .all(|&v| v == marker(4096, head))
                );
                let first = (head as usize * 4097) * 128;
                assert!(
                    actual[first..first + 128]
                        .iter()
                        .all(|&v| v == marker(0, head))
                );
            }
        }
        Plane::commit_chunk_chained(&mut k, &mut state.planes[0], guard, &keys, &values).unwrap();
        // Retire the accumulated graph like the D2 driver does (the plane
        // is back in `held` after the chained commit).
        if pos % 512 == 0 {
            let roots = state.planes[0].as_ref().unwrap().append_roots();
            k.evaluate(&roots).unwrap();
        }
        state.pos = pos + 1;
    }

    // SWA layer 2: bounded union regardless of the flag (rope layers never
    // take the ring arm).
    let mut swa = State::new();
    for pos in 0u32..=600 {
        swa.ensure_full_capacity(&mut k, pos + 1).unwrap();
        let plan = ChunkCachePlan::new(2, pos, 1, swa.full_capacity).unwrap();
        let row = row_for(&mut k, pos);
        let (keys, values, guard) =
            Plane::prepare_chunk(&mut k, &mut swa.planes[2], &plan, &row, &row, true).unwrap();
        assert_eq!(
            keys.shape(),
            [4, (pos + 1).min(WINDOW), 128],
            "SWA at pos {pos}"
        );
        Plane::commit_chunk_chained(&mut k, &mut swa.planes[2], guard, &keys, &values).unwrap();
        swa.pos = pos + 1;
    }

    // Multi-row chunks (prefill) never meet the ring arm: the union path
    // serves exactly the chunk rows then the grown history.
    let mut prefill = State::new();
    prefill.ensure_full_capacity(&mut k, 4).unwrap();
    let chunk = |k: &mut MlxKernels, pos: u32, rows: u32| {
        let values: Vec<f32> = (0..4)
            .flat_map(|head| (0..rows).flat_map(move |r| vec![marker(pos + r, head); 128]))
            .collect();
        let host = Tensor::try_from_host(
            Dtype::F32,
            vec![4, rows, 128],
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
        k.import_tensor(&host).unwrap()
    };
    for (p0, rows, want) in [(0u32, 4u32, 4u32), (4, 4, 8)] {
        prefill.ensure_full_capacity(&mut k, p0 + rows).unwrap();
        let plan = ChunkCachePlan::new(0, p0, rows, prefill.full_capacity).unwrap();
        let block = chunk(&mut k, p0, rows);
        let (keys, values, guard) =
            Plane::prepare_chunk(&mut k, &mut prefill.planes[0], &plan, &block, &block, true)
                .unwrap();
        assert_eq!(keys.shape(), [4, want, 128], "prefill chunk {p0}+{rows}");
        Plane::commit_chunk_chained(&mut k, &mut prefill.planes[0], guard, &keys, &values).unwrap();
        prefill.pos = p0 + rows;
    }
}
