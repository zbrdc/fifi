use super::*;

mod allocation;
mod sink_decode;

const HEADS: usize = 32;
const KV_HEADS: usize = 4;
const DIM: usize = 128;
const WINDOW: usize = 513;

// Stage chronological logical rows into a genuinely bounded physical allocation.
fn staged_plane(shape: AttnPrefillShape, role: usize, origin: usize) -> (Plane, Vec<f64>) {
    let storage = AttnKvStorage::Q8Affine {
        group_size: 32,
        signed: true,
    };
    let logical = plane(shape, storage, (Dtype::F16, None), role);
    let mut physical = plane(shape, storage, (Dtype::F16, None), role);
    let scalar = logical.layout.scales.unwrap();
    let layouts = [
        (0, logical.layout.payload_strides, DIM),
        (
            1,
            scalar.strides,
            (DIM / 32 - 1) * scalar.group_stride_bytes + 2,
        ),
    ];
    for row in 0..shape.cache_rows {
        let target = (origin + row) % shape.cache_rows;
        for h in 0..KV_HEADS {
            for (index, strides, bytes) in layouts {
                let from = ORIGINS[index]
                    + row * strides.time_stride_bytes
                    + h * strides.head_stride_bytes;
                let to = ORIGINS[index]
                    + target * strides.time_stride_bytes
                    + h * strides.head_stride_bytes;
                physical.bytes[index][to..to + bytes]
                    .copy_from_slice(&logical.bytes[index][from..from + bytes]);
            }
            let from = (row * KV_HEADS + h) * DIM;
            let to = (target * KV_HEADS + h) * DIM;
            physical.values[to..to + DIM].copy_from_slice(&logical.values[from..from + DIM]);
        }
    }
    (physical, logical.values)
}

// With Q=0 every visible key has weight one. This independently checks masks and raw sinks,
// not nonzero QK arithmetic or model projections (covered by the existing numerical tests).
fn zero_q_reference(
    shape: AttnPrefillShape,
    span: CpfAttnSpan,
    values: &[f64],
    sinks: &[f32],
) -> Vec<f64> {
    let width = KV_HEADS * DIM;
    let origin = if span.ring > 0 {
        (shape.p0 as usize).saturating_sub(WINDOW - 1)
    } else {
        0
    };
    let mut prefix = vec![0.0; (shape.cache_rows + 1) * width];
    for row in 0..shape.cache_rows {
        for i in 0..width {
            prefix[(row + 1) * width + i] = prefix[row * width + i] + values[row * width + i];
        }
    }
    let mut out = vec![0.0; shape.m * HEADS * DIM];
    for t in 0..shape.m {
        let end = shape.p0 as usize + t + 1;
        let first = if span.window > 0 {
            end.saturating_sub(span.window as usize)
        } else {
            0
        };
        for (h, sink) in sinks.iter().enumerate() {
            let denominator = (end - first) as f64 + f64::from(*sink).exp();
            for d in 0..DIM {
                let lane = (h / (HEADS / KV_HEADS)) * DIM + d;
                let sum =
                    prefix[(end - origin) * width + lane] - prefix[(first - origin) * width + lane];
                out[(t * HEADS + h) * DIM + d] = sum / denominator;
            }
        }
    }
    out
}

struct BlockObservation {
    actual: Vec<f32>,
    reference: Vec<f64>,
    status: i32,
    guards: bool,
    refusals: bool,
}

fn observe_block(
    m: usize,
    p0: usize,
    span: CpfAttnSpan,
    storage: AttnKvStorage,
) -> Result<BlockObservation, EngError> {
    let shape = AttnPrefillShape {
        m,
        p0: p0 as i64,
        n_head: HEADS,
        n_kv_head: KV_HEADS,
        head_dim: DIM,
        ld: 0,
        cache_rows: if span.ring > 0 {
            span.ring as usize
        } else {
            p0 + m
        },
    };
    let origin = if span.ring > 0 {
        p0.saturating_sub(WINDOW - 1)
    } else {
        0
    };
    let (k, _) = staged_plane(shape, 0, origin);
    let (v, logical_v) = staged_plane(shape, 1, origin);
    let input = Data {
        q: vec![0.0; m * HEADS * DIM],
        k: k.values.iter().map(|&x| f16_bits(x as f32)).collect(),
        v: v.values.iter().map(|&x| f16_bits(x as f32)).collect(),
        proj: Vec::new(),
    };
    let split = CpfAttnSplit {
        max_splits: 3,
        min_keys: 32,
    };
    let bytes = |rows| {
        cpf_attn_part_bytes(
            CpfAttnGeometry {
                m: rows,
                n_head: HEADS,
                n_kv_head: KV_HEADS,
                head_dim: DIM,
            },
            CpfAttnPlan {
                q_tile: CPF_ATTN_Q_TILE,
                max_splits: split.max_splits,
                min_keys: split.min_keys,
            },
        )
    };
    let mut tile = m;
    while bytes(tile)? > CPF_ATTN_PART_CAP {
        tile /= 2;
    }
    let mut fx = Fx::with_data(
        AttnPrefillShape { m: tile, ..shape },
        split,
        false,
        true,
        &input,
    )?;
    fx.span = span;
    fx.scale = 0.125;
    let sinks: Vec<f32> = (0..HEADS).map(|h| h as f32 / 4.0 - 4.0).collect();
    fx.sink_buf = Some(fx.ctx.upload(&le(&sinks))?);
    let full_bytes = input.q.len() * 4;
    fx.bufs[OUT] = testkit::upload_filled(&fx.ctx, OUT_FILL, full_bytes + SLACK);
    let quantized = storage != AttnKvStorage::F16;
    let buffers = if quantized {
        Some([k.upload(&fx.ctx)?, v.upload(&fx.ctx)?])
    } else {
        None
    };
    let views = match &buffers {
        Some([kb, vb]) => Some([k.view(kb)?, v.view(vb)?]),
        None => None,
    };
    let call = |offset: usize, count: usize| -> Result<AttnPrefillCall<'_>, EngError> {
        let mut call = fx.call()?;
        call.shape = AttnPrefillShape {
            m: count,
            p0: (p0 + offset) as i64,
            ..shape
        };
        call.q = fx.bufs[Q].range(offset * HEADS * DIM * 4, count * HEADS * DIM * 4)?;
        call.out = fx.bufs[OUT].range(offset * HEADS * DIM * 4, count * HEADS * DIM * 4)?;
        if let Some(views) = views {
            call.storage = storage;
            call.k_cache = views[0].payload;
            call.v_cache = views[1].payload;
            call.kv_views = Some(views);
        }
        Ok(call)
    };
    let before = [fx.bytes(PART), fx.bytes(OUT), fx.bytes(STATUS)];
    let mut refusals = true;
    if span.ring > 0 && m > 1 {
        let mut bad = call(0, tile)?;
        bad.span.ring = WINDOW as i64;
        refusals &= matches!(
            fx.ctx.attn_prefill(&fx.stream, bad),
            Err(EngError::BadRequest(_))
        );
    }
    if tile < m {
        refusals &= matches!(
            fx.ctx.attn_prefill(&fx.stream, call(0, m)?),
            Err(EngError::MemoryPlan(_))
        );
    }
    refusals &= before == [fx.bytes(PART), fx.bytes(OUT), fx.bytes(STATUS)];
    for offset in (0..m).step_by(tile) {
        fx.ctx
            .attn_prefill(&fx.stream, call(offset, tile.min(m - offset))?)?;
    }
    fx.stream.synchronize()?;
    let actual = testkit::floats(&fx.bytes(OUT)[..full_bytes]);
    let status = fx.status_word();
    let guards = fx.bytes(OUT)[full_bytes..].iter().all(|&b| b == OUT_FILL)
        && fx.bytes(PART)[fx.base[PART].2..]
            .iter()
            .all(|&b| b == PART_FILL)
        && fx.bytes(Q)[..full_bytes] == le(&input.q)
        && fx.bytes(KC)[..input.k.len() * 2] == le16(&input.k)
        && fx.bytes(VC)[..input.v.len() * 2] == le16(&input.v)
        && fx.bytes(STATUS)[4..].iter().all(|&b| b == 0)
        && fx
            .sink_buf
            .as_ref()
            .is_some_and(|b| testkit::read_bytes(&fx.ctx, b, b.bytes()) == le(&sinks))
        && buffers
            .as_ref()
            .is_none_or(|[kb, vb]| k.unchanged(&fx.ctx, kb) && v.unchanged(&fx.ctx, vb));
    drop((buffers, fx));
    let reference = zero_q_reference(shape, span, &logical_v, &sinks);
    Ok(BlockObservation {
        actual,
        reference,
        status,
        guards,
        refusals,
    })
}

#[test]
fn flint_f16_then_q8_blocks_match_physical_ring_mask_and_sink_oracle() {
    if !testkit::device_present("flint_f16_then_q8_blocks_match_physical_ring_mask_and_sink_oracle")
    {
        return;
    }
    let rows = [
        1, 2, 7, 8, 9, 31, 32, 33, 511, 512, 513, 514, 1023, 1024, 1025, 2048, 4096,
    ];
    let mut cases = Vec::new();
    for m in rows {
        cases.push((m, 0, CpfAttnSpan::FULL));
        cases.push((
            m,
            1026,
            CpfAttnSpan {
                window: WINDOW as i64,
                ring: (WINDOW + m - 1) as i64,
            },
        ));
    }
    for p0 in [0, 511, 512, 513] {
        cases.push((
            17,
            p0,
            CpfAttnSpan {
                window: WINDOW as i64,
                ring: 0,
            },
        ));
    }
    cases.push((1, 1026, CpfAttnSpan::FULL));
    cases.push((
        1,
        ((1_usize << 30) - 2),
        CpfAttnSpan {
            window: WINDOW as i64,
            ring: WINDOW as i64,
        },
    ));
    for storage in [
        AttnKvStorage::F16,
        AttnKvStorage::Q8Affine {
            group_size: 32,
            signed: true,
        },
    ] {
        for &(m, p0, span) in &cases {
            let observed = observe_block(m, p0, span, storage);
            let observed = observed.expect("device owners dropped before assertions");
            assert_eq!(observed.status, 0);
            assert!(observed.guards && observed.refusals);
            assert_eq!(observed.actual.len(), observed.reference.len());
            let mut worst_ratio = 0.0_f64;
            for (got, want) in observed
                .actual
                .as_chunks::<DIM>()
                .0
                .iter()
                .zip(observed.reference.as_chunks::<DIM>().0)
            {
                let magnitude = want.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                let exponent = ((magnitude.to_bits() >> 52) & 0x7ff) as i32 - 1023;
                let bound = 2.0 * 2.0_f64.powi(exponent.max(-126) - 7);
                for (&actual, &expected) in got.iter().zip(want) {
                    assert!(actual.is_finite() && expected.is_finite());
                    worst_ratio = worst_ratio.max((f64::from(actual) - expected).abs() / bound);
                }
            }
            assert!(
                worst_ratio <= 1.0,
                "{storage:?} M{m} P{p0} {span:?}: error/bound={worst_ratio}"
            );
            eprintln!(
                "Flint shared correctness: {storage:?} M{m} P{p0} {span:?}; error/bound={worst_ratio}"
            );
        }
    }
}
