use super::{
    AttnKvStorage, AttnPrefillShape, CPF_ATTN_Q_TILE, CpfAttnSpan, MAX_ROWS, check_full_shape,
    check_storage,
};

#[test]
fn flint_physical_ring_shape_admits_deep_positions_without_relaxing_linear_bounds() {
    let mut shape = AttnPrefillShape {
        m: 1,
        n_head: 32,
        n_kv_head: 4,
        head_dim: 128,
        cache_rows: 513,
        ld: 0,
        p0: 1026,
    };
    let ring = CpfAttnSpan {
        window: 513,
        ring: 513,
    };
    for storage in [AttnKvStorage::F16, quantized(32)[0]] {
        let admitted = check_full_shape(&shape, CPF_ATTN_Q_TILE, &ring, storage);
        assert!(admitted.is_ok(), "{storage:?}: {:?}", admitted.err());
        assert!(check_full_shape(&shape, CPF_ATTN_Q_TILE, &CpfAttnSpan::FULL, storage).is_err());
        shape.p0 = MAX_ROWS as i64 - 1;
        assert!(check_full_shape(&shape, CPF_ATTN_Q_TILE, &ring, storage).is_ok());
        for p0 in [-1, MAX_ROWS as i64, i64::MAX] {
            shape.p0 = p0;
            assert!(check_full_shape(&shape, CPF_ATTN_Q_TILE, &ring, storage).is_err());
        }
        shape.p0 = 1026;
    }
}

pub(super) fn quantized(group_size: usize) -> [AttnKvStorage; 4] {
    [
        AttnKvStorage::Q8Affine {
            group_size,
            signed: true,
        },
        AttnKvStorage::Q8Affine {
            group_size,
            signed: false,
        },
        AttnKvStorage::Q4Affine {
            group_size,
            low_nibble_first: true,
        },
        AttnKvStorage::Q4Affine {
            group_size,
            low_nibble_first: false,
        },
    ]
}

#[test]
fn flint_every_query_count_uses_bounded_subtiles_without_raising_the_part_cap() {
    use super::{
        CPF_ATTN_PART_CAP, CPF_ATTN_SM_COUNT, CpfAttnGeometry, CpfAttnPlan, cpf_attn_part_bytes,
        cpf_attn_split_plan,
    };
    let geometry = |m| CpfAttnGeometry {
        m,
        n_head: 32,
        n_kv_head: 4,
        head_dim: 128,
    };
    let plan = CpfAttnPlan {
        q_tile: CPF_ATTN_Q_TILE,
        max_splits: 3,
        min_keys: 32,
    };
    let one_split = CpfAttnPlan {
        max_splits: 1,
        ..plan
    };
    assert_eq!(
        cpf_attn_part_bytes(geometry(4096), one_split).unwrap(),
        68_157_440
    );
    assert!(matches!(
        cpf_attn_split_plan(geometry(4096), 20_480, CPF_ATTN_Q_TILE, CPF_ATTN_SM_COUNT),
        Err(crate::EngError::MemoryPlan(_))
    ));
    for m in 1..=4096 {
        let shape = AttnPrefillShape {
            m,
            n_head: 32,
            n_kv_head: 4,
            head_dim: 128,
            cache_rows: 512 + m,
            ld: 0,
            p0: 16_384,
        };
        let span = CpfAttnSpan {
            window: 513,
            ring: shape.cache_rows as i64,
        };
        for storage in [AttnKvStorage::F16, quantized(32)[0]] {
            assert!(check_full_shape(&shape, CPF_ATTN_Q_TILE, &span, storage).is_ok());
            assert!(super::check_shape(&shape, CPF_ATTN_Q_TILE).is_err());
        }
        let mut tile = m;
        while cpf_attn_part_bytes(geometry(tile), plan).unwrap() > CPF_ATTN_PART_CAP {
            tile /= 2;
        }
        assert!(tile > 1 || m == 1);
        let mut covered = 0;
        for offset in (0..m).step_by(tile) {
            let rows = tile.min(m - offset);
            assert!(cpf_attn_part_bytes(geometry(rows), plan).unwrap() <= CPF_ATTN_PART_CAP);
            assert!(span.ring >= span.window + rows as i64 - 1);
            covered += rows;
        }
        assert_eq!(covered, m);
        if m == 4096 {
            assert_eq!(tile, 1024);
        }
    }
}

#[test]
fn storage_payload_sizes_cover_the_format_and_group_contract() {
    for dim in [64, 128, 256] {
        assert_eq!(AttnKvStorage::F16.payload_cell_bytes(dim), Some(dim * 2));
        assert_eq!(AttnKvStorage::Bf16.payload_cell_bytes(dim), Some(dim * 2));
        assert_eq!(AttnKvStorage::F32.payload_cell_bytes(dim), Some(dim * 4));
        for group in [16, 32, 64, 128, 256] {
            let valid = dim % group == 0;
            for (index, storage) in quantized(group).into_iter().enumerate() {
                let bytes = if index < 2 { dim } else { dim / 2 };
                assert_eq!(storage.payload_cell_bytes(dim), valid.then_some(bytes));
            }
        }
    }
}

#[test]
fn storage_payload_sizes_reject_empty_overflow_and_invalid_groups() {
    for storage in [AttnKvStorage::F16, AttnKvStorage::Bf16, AttnKvStorage::F32] {
        assert_eq!(storage.payload_cell_bytes(0), None);
        assert_eq!(storage.payload_cell_bytes(usize::MAX), None);
    }
    for group in [0, 17, 65, usize::MAX] {
        for storage in quantized(group) {
            assert_eq!(storage.payload_cell_bytes(64), None);
        }
    }
    for storage in quantized(1) {
        assert_eq!(storage.payload_cell_bytes(0), None);
    }
    assert_eq!(
        quantized(1)[2].payload_cell_bytes(usize::MAX),
        Some(usize::MAX / 2 + 1)
    );
}

#[test]
fn q8_affine_storage_admits_each_dim_group_and_signedness() {
    for dim in [64, 128, 256] {
        for group in [16, 32, 64, 128, 256] {
            if dim % group != 0 {
                continue;
            }
            for storage in quantized(group).into_iter().take(2) {
                assert!(check_storage(storage, dim).is_ok(), "D{dim}/{storage:?}");
            }
        }
    }
}

#[test]
fn q8_shape_admission_covers_runtime_gqa_and_rejects_zero_heads() {
    use super::{
        AttnPrefillShape, CPF_ATTN_Q_TILE, CpfAttnGeometry, CpfAttnPlan, check_shape_mode,
        cpf_attn_part_bytes,
    };
    let storage = AttnKvStorage::Q8Affine {
        group_size: 32,
        signed: true,
    };
    for head_dim in [64, 128, 256] {
        for group in [1, 3, 4, 8, 16] {
            let mut shape = AttnPrefillShape {
                m: 17,
                n_head: 2 * group,
                n_kv_head: 2,
                head_dim,
                cache_rows: 64,
                ld: 0,
                p0: 31,
            };
            assert!(check_shape_mode(&shape, CPF_ATTN_Q_TILE, false, storage).is_ok());
            shape.n_head = 0;
            assert!(check_shape_mode(&shape, CPF_ATTN_Q_TILE, false, storage).is_err());
            let geometry = CpfAttnGeometry {
                m: 1,
                n_head: 0,
                n_kv_head: 1,
                head_dim,
            };
            let plan = CpfAttnPlan {
                q_tile: 8,
                max_splits: 1,
                min_keys: 32,
            };
            assert!(cpf_attn_part_bytes(geometry, plan).is_err());
        }
    }
}

#[test]
fn undeveloped_formats_are_refused_instead_of_reinterpreted_as_f16() {
    for dim in [64, 128, 256] {
        assert!(check_storage(AttnKvStorage::F16, dim).is_ok());
        for storage in [AttnKvStorage::Bf16, AttnKvStorage::F32]
            .into_iter()
            .chain(quantized(16).into_iter().skip(2))
        {
            let error = format!("{:?}", check_storage(storage, dim).unwrap_err());
            assert!(error.contains("attn_prefill: storage"), "{error}");
            assert!(error.contains("not implemented"), "{error}");
        }
    }
    for dim in [32, 512] {
        for storage in quantized(16).into_iter().take(2) {
            assert!(check_storage(storage, dim).is_err());
        }
    }
    for storage in quantized(0) {
        let error = format!("{:?}", check_storage(storage, 64).unwrap_err());
        assert!(error.contains("group"), "{error}");
    }
}
