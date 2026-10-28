use super::*;
use crate::family::flint::{cuda_kv::LayerKvAllocation, kv_plan::KvPlan};

const CONTEXT: usize = 8192;

struct History {
    shape: AttnPrefillShape,
    q: Vec<f32>,
    k: Vec<f64>,
    v: Vec<f64>,
    sinks: Vec<f32>,
}

fn history(m: usize) -> History {
    let prefix = if m == 512 { CONTEXT - WINDOW } else { CONTEXT };
    let shape = AttnPrefillShape {
        m,
        p0: (prefix - m) as i64,
        n_head: HEADS,
        n_kv_head: KV_HEADS,
        head_dim: DIM,
        ld: 0,
        cache_rows: CONTEXT,
    };
    let storage = AttnKvStorage::Q8Affine {
        group_size: 32,
        signed: true,
    };
    History {
        shape,
        q: (0..m * HEADS * DIM)
            .map(|i| {
                if m == 1 {
                    ((i % 31) as f32 - 15.0) / 64.0
                } else {
                    0.0
                }
            })
            .collect(),
        k: plane(shape, storage, (Dtype::F16, None), 0).values,
        v: plane(shape, storage, (Dtype::F16, None), 1).values,
        sinks: (0..HEADS).map(|h| h as f32 / 4.0 - 4.0).collect(),
    }
}

fn physical_plane(logical: &[f64], rows: usize, end: usize) -> Vec<u16> {
    let width = KV_HEADS * DIM;
    let mut physical = vec![F16_NAN; rows * width];
    for position in end.saturating_sub(rows)..end {
        for lane in 0..width {
            physical[(position % rows) * width + lane] =
                f16_bits(logical[position * width + lane] as f32);
        }
    }
    physical
}

fn launch_rows(m: usize, split: CpfAttnSplit) -> Result<usize, EngError> {
    let mut rows = m;
    loop {
        let bytes = cpf_attn_part_bytes(
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
        )?;
        if bytes <= CPF_ATTN_PART_CAP {
            return Ok(rows);
        }
        rows /= 2;
    }
}

struct AllocatedObservation {
    actual: Vec<f32>,
    rows: usize,
    bytes: usize,
    status: i32,
    allocation: bool,
    guards: bool,
}

fn observe_allocated(
    input: &History,
    plan: &KvPlan,
    layer: usize,
) -> Result<AllocatedObservation, EngError> {
    let allocation = LayerKvAllocation::new(plan, layer)?;
    let shape = AttnPrefillShape {
        cache_rows: allocation.cache_rows(),
        ..input.shape
    };
    let span = if layer == 0 {
        CpfAttnSpan::FULL
    } else {
        CpfAttnSpan {
            window: WINDOW as i64,
            ring: if shape.cache_rows < CONTEXT {
                shape.cache_rows as i64
            } else {
                0
            },
        }
    };
    let end = shape.p0 as usize + shape.m;
    let data = Data {
        q: input.q.clone(),
        k: physical_plane(&input.k, shape.cache_rows, end),
        v: physical_plane(&input.v, shape.cache_rows, end),
        proj: Vec::new(),
    };
    let split = CpfAttnSplit {
        max_splits: 3,
        min_keys: 32,
    };
    // Tile only the launches: allocation must retain the entire original chunk union.
    let tile = launch_rows(shape.m, split)?;
    let mut fx = Fx::with_data(
        AttnPrefillShape { m: tile, ..shape },
        split,
        false,
        true,
        &data,
    )?;
    fx.span = span;
    fx.scale = 0.125;
    fx.sink_buf = Some(fx.ctx.upload(&le(&input.sinks))?);
    let full_bytes = input.q.len() * 4;
    fx.bufs[OUT] = testkit::upload_filled(&fx.ctx, OUT_FILL, full_bytes + SLACK);
    let before = fx.ctx.live_bytes();
    let owned = allocation.allocate(&fx.ctx)?;
    let bytes = owned.keys.bytes() + owned.values.bytes();
    let allocation_ok = bytes == 2 * allocation.plane_bytes()
        && owned.keys.bytes() == allocation.plane_bytes()
        && owned.values.bytes() == allocation.plane_bytes()
        && fx.ctx.live_bytes() - before == bytes;
    owned.keys.write_at(&fx.stream, 0, &le16(&data.k))?;
    owned.values.write_at(&fx.stream, 0, &le16(&data.v))?;
    // Replace the fixture caches; every launch now reads the actual allocator owners.
    fx.bufs[KC] = owned.keys;
    fx.bufs[VC] = owned.values;
    for offset in (0..shape.m).step_by(tile) {
        let count = tile.min(shape.m - offset);
        let row_bytes = HEADS * DIM * 4;
        let mut call = fx.call()?;
        call.shape = AttnPrefillShape {
            m: count,
            p0: shape.p0 + offset as i64,
            ..shape
        };
        call.q = fx.bufs[Q].range(offset * row_bytes, count * row_bytes)?;
        call.out = fx.bufs[OUT].range(offset * row_bytes, count * row_bytes)?;
        fx.ctx.attn_prefill(&fx.stream, call)?;
    }
    fx.stream.synchronize()?;
    let actual = testkit::floats(&fx.bytes(OUT)[..full_bytes]);
    let status = fx.status_word();
    let guards = fx.bytes(OUT)[full_bytes..].iter().all(|&b| b == OUT_FILL)
        && fx.bytes(PART)[fx.base[PART].2..]
            .iter()
            .all(|&b| b == PART_FILL)
        && fx.bytes(Q)[..full_bytes] == le(&input.q)
        && fx.bytes(KC) == le16(&data.k)
        && fx.bytes(VC) == le16(&data.v)
        && fx.bytes(STATUS)[4..].iter().all(|&b| b == 0)
        && fx
            .sink_buf
            .as_ref()
            .is_some_and(|b| testkit::read_bytes(&fx.ctx, b, b.bytes()) == le(&input.sinks));
    drop(fx);
    Ok(AllocatedObservation {
        actual,
        rows: shape.cache_rows,
        bytes,
        status,
        allocation: allocation_ok,
        guards,
    })
}

fn reference(input: &History, layer: usize) -> Vec<f64> {
    let window = if layer == 0 { 0 } else { WINDOW };
    if input.shape.m != 1 {
        return zero_q_reference(
            input.shape,
            CpfAttnSpan {
                window: window as i64,
                ring: 0,
            },
            &input.v,
            &input.sinks,
        );
    }
    let q: Vec<f64> = input.q.iter().copied().map(f64::from).collect();
    let sinks: Vec<f64> = input.sinks.iter().copied().map(f64::from).collect();
    sdpa_f64(
        &q,
        &input.k,
        &input.v,
        input.shape.m,
        HEADS,
        CONTEXT,
        DIM,
        HEADS / KV_HEADS,
        0.125,
        input.shape.p0 as u32,
        0,
        (window > 0).then_some(window as u32),
        Some(&sinks),
    )
}

pub(super) fn check_close(actual: &[f32], reference: &[f64]) -> f64 {
    assert_eq!(actual.len(), reference.len());
    let mut worst = 0.0_f64;
    for (got, want) in actual
        .as_chunks::<DIM>()
        .0
        .iter()
        .zip(reference.as_chunks::<DIM>().0)
    {
        let magnitude = want.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let exponent = ((magnitude.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        let bound = 2.0 * 2.0_f64.powi(exponent.max(-126) - 7);
        for (&actual, &expected) in got.iter().zip(want) {
            assert!(actual.is_finite() && expected.is_finite());
            worst = worst.max((f64::from(actual) - expected).abs() / bound);
        }
    }
    assert!(worst <= 1.0, "error/bound={worst}");
    worst
}

#[test]
#[ignore = "requires an admitted CUDA correctness run; never silently skips a missing device"]
fn flint_kv_plan_allocations_match_full_cache_at_depth() {
    for m in [1, 4, 512] {
        let input = history(m);
        for layer in [0, 2] {
            let expected = reference(&input, layer);
            let full_plan = KvPlan::new(CONTEXT, m, false).unwrap();
            let ring_plan = KvPlan::new(CONTEXT, m, true).unwrap();
            let full = observe_allocated(&input, &full_plan, layer).expect("required CUDA device");
            let ring = observe_allocated(&input, &ring_plan, layer).expect("required CUDA device");
            for observed in [&full, &ring] {
                assert!(observed.allocation && observed.guards);
                assert_eq!(observed.status, 0);
                let ratio = check_close(&observed.actual, &expected);
                eprintln!(
                    "Flint allocated correctness M{m} P{} layer{layer} rows{} bytes{} error/bound={ratio}",
                    input.shape.p0, observed.rows, observed.bytes
                );
            }
            let full_values: Vec<f64> = full.actual.iter().copied().map(f64::from).collect();
            check_close(&ring.actual, &full_values);
            assert_eq!(full.rows, CONTEXT);
            assert_eq!(ring.rows, if layer == 0 { CONTEXT } else { WINDOW + m - 1 });
        }
    }
}
