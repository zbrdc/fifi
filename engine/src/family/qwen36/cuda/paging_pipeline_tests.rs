//! Real KV8 producer -> host writeback/reload -> checked attention, not model qualification.
use super::{
    common_tests::dims,
    native_device_rig::{device_present, f16},
    paging_plan::PagingPlan,
    paging_store::{PageAttention, PageChunk, PageId, PagedKv},
    state_layout::{KvPool, Plane},
};
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream,
        resident::{
            attn_decode::{AttnSpan, AttnSplit},
            attn_decode_q8::{AttnDecodeQ8Call, AttnQ8Shape, attn_q8_part_bytes},
            kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape},
        },
    },
};

const N: usize = 257;
const C: usize = 256;
const D: usize = 256;
const KV: usize = 2;
const HEADS: usize = KV * 8;
const ABS: f64 = 1e-5;
const E3: f64 = 1.0 / 131_072.0;

pub(super) fn upload(
    ctx: &CudaContext,
    stream: &DeviceStream,
    bytes: &[u8],
) -> Result<DeviceBuffer, EngError> {
    let device = ctx.allocate(bytes.len())?;
    let mut pinned = ctx.pinned_buf(bytes.len())?;
    let event = ctx.event()?;
    pinned.write(0, bytes)?;
    ctx.copy_from_pinned(stream, &mut pinned, 0, device.range(0, bytes.len())?)?;
    stream.record(&event)?;
    pinned.wait(&event)?;
    Ok(device)
}

pub(super) fn read(
    ctx: &CudaContext,
    stream: &DeviceStream,
    range: DeviceRange<'_>,
) -> Result<Vec<u8>, EngError> {
    let mut pinned = ctx.pinned_buf(range.bytes())?;
    let event = ctx.event()?;
    ctx.copy_to_pinned(stream, range, &mut pinned, 0)?;
    stream.record(&event)?;
    pinned.wait(&event)?;
    let mut bytes = vec![0; range.bytes()];
    pinned.read(0, &mut bytes)?;
    Ok(bytes)
}

pub(super) fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub(super) fn inputs(rows: usize) -> (Vec<f32>, Vec<f32>) {
    let k = (0..rows * KV * D)
        .map(|i| {
            let (t, h, d) = (i / (KV * D), i / D % KV, i % D);
            ((t * 7 + h * 11 + d) % 61) as f32 / 16.0 - 30.0 / 16.0
        })
        .collect();
    let v = (0..rows * KV * D)
        .map(|i| {
            let (t, h, d) = (i / (KV * D), i / D % KV, i % D);
            ((t * 13 + h * 17 + d * 3) % 83) as f32 / 32.0 - 41.0 / 32.0
        })
        .collect();
    (k, v)
}

pub(super) fn query_and_gate() -> (Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0; HEADS * D];
    let mut gate = vec![f32::NAN; HEADS * 2 * D];
    for h in 0..HEADS {
        q[h * D..h * D + 4].copy_from_slice(&[0.125 * (h + 1) as f32, -0.25, 0.5, 0.125]);
        for d in 0..D {
            gate[h * 2 * D + D + d] = ((h + d) % 19) as f32 / 16.0 - 0.5;
        }
    }
    (q, gate)
}

pub(super) fn plane<'a>(
    buffer: &'a DeviceBuffer,
    pool: &KvPool,
    p: Plane,
) -> Result<DeviceRange<'a>, EngError> {
    let (offset, bytes) = pool.plane_extent(0, p)?;
    buffer.range(offset, bytes)
}

struct Observation {
    reference: Vec<u8>,
    pages: [Vec<u8>; 2],
    dense: Vec<u8>,
    paged: Vec<u8>,
    isolated: Vec<u8>,
    pool: KvPool,
    group: KvPool,
}

fn observe() -> Result<Observation, EngError> {
    let dimensions = dims();
    let plan = PagingPlan::new(&dimensions, 320)?;
    let pool = KvPool::q8_staging_layer(&dimensions, N)?;
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let (k, v) = inputs(N);
    let k = upload(&ctx, &stream, &floats(&k))?;
    let v = upload(&ctx, &stream, &floats(&v))?;
    let reference = upload(&ctx, &stream, &vec![0; pool.total])?;
    ctx.kv8_chunk(
        &stream,
        Kv8Shape {
            rows: N,
            n_kv_head: KV,
            head_dim: D,
            cells: N,
            code_scale: Kv8CodeScale::StoredHalf,
        },
        Kv8Data {
            k: k.range(0, k.bytes())?,
            v: v.range(0, v.bytes())?,
            k_codes: plane(&reference, &pool, Plane::K)?,
            k_scales: plane(&reference, &pool, Plane::KScales)?,
            v_codes: plane(&reference, &pool, Plane::V)?,
            v_scales: plane(&reference, &pool, Plane::VScales)?,
        },
        0,
    )?;
    let expected = read(&ctx, &stream, reference.range(0, reference.bytes())?)?;
    for p in pool.planes() {
        let (start, len) = pool.plane_extent(0, *p)?;
        if expected[start..start + len].iter().all(|b| *b == 0) {
            return Err(EngError::BadRequest(
                "dense KV8 producer control is empty".to_owned(),
            ));
        }
    }
    eprintln!("HOST_PAGE_DENSE_PRODUCER_CONTROL_PASS");
    let mut store = PagedKv::allocate(&ctx, &dimensions, 320, 8 << 20)?;
    for group in 0..2 {
        let page = PageId {
            full_layer: 9,
            group,
        };
        store.load(&ctx, &stream, group, page)?;
        let rows = if group == 0 { C } else { 1 };
        let offset = group * C * KV * D * 4;
        eprintln!("HOST_PAGE_APPEND_REAL_DEVICE_READY group={group}");
        store.append_chunk(
            &ctx,
            &stream,
            group,
            PageChunk {
                page,
                position: group * C,
                rows,
                k: k.range(offset, rows * KV * D * 4)?,
                v: v.range(offset, rows * KV * D * 4)?,
            },
        )?;
        // This fence, not a test readback/synchronize, must cover the queued KV8 writer.
        store.writeback(&ctx, &stream, group)?;
    }
    let mut pages = [Vec::new(), Vec::new()];
    for (group, bytes) in pages.iter_mut().enumerate() {
        let slot = 1 - group;
        store.load(
            &ctx,
            &stream,
            slot,
            PageId {
                full_layer: 9,
                group,
            },
        )?;
        *bytes = read(&ctx, &stream, store.resident(slot)?)?;
        store.writeback(&ctx, &stream, slot)?;
    }
    let (query, gate) = query_and_gate();
    let q = upload(&ctx, &stream, &floats(&query))?;
    let proj = upload(&ctx, &stream, &floats(&gate))?;
    let len = upload(&ctx, &stream, &(N as i32).to_le_bytes())?;
    let part = ctx.allocate(attn_q8_part_bytes(1, KV, D, 64)?)?;
    let dense = ctx.allocate(HEADS * D * 4)?;
    let paged = ctx.allocate(HEADS * D * 4)?;
    let shape = AttnQ8Shape {
        m: 1,
        n_head: HEADS,
        n_kv_head: KV,
        head_dim: D,
        cells: N,
        ld: HEADS * 2 * D,
    };
    ctx.attn_decode_q8(
        &stream,
        AttnDecodeQ8Call {
            shape,
            span: AttnSpan::FULL,
            split: AttnSplit {
                max_splits: 64,
                min_keys: C,
            },
            scale: 1.0 / 16.0,
            q: q.range(0, q.bytes())?,
            proj: Some(proj.range(0, proj.bytes())?),
            k_codes: plane(&reference, &pool, Plane::K)?,
            k_scales: plane(&reference, &pool, Plane::KScales)?,
            v_codes: plane(&reference, &pool, Plane::V)?,
            v_scales: plane(&reference, &pool, Plane::VScales)?,
            kv_len: len.range(0, 4)?,
            part: part.range(0, part.bytes())?,
            out: dense.range(0, dense.bytes())?,
        },
    )?;
    let dense = read(&ctx, &stream, dense.range(0, dense.bytes())?)?;
    let mut scratch = ctx.paged_q8_scratch(
        AttnQ8Shape {
            cells: 320,
            ..shape
        },
        C,
    )?;
    store.attend(
        &ctx,
        &stream,
        &mut scratch,
        PageAttention {
            full_layer: 9,
            visible_keys: N,
            scale: 1.0 / 16.0,
            q: q.range(0, q.bytes())?,
            proj: Some(proj.range(0, proj.bytes())?),
            out: paged.range(0, paged.bytes())?,
        },
    )?;
    let paged = read(&ctx, &stream, paged.range(0, paged.bytes())?)?;
    store.load(
        &ctx,
        &stream,
        0,
        PageId {
            full_layer: 0,
            group: 0,
        },
    )?;
    let isolated = read(&ctx, &stream, store.resident(0)?)?;
    Ok(Observation {
        reference: expected,
        pages,
        dense,
        paged,
        isolated,
        pool,
        group: plan.group,
    })
}

fn packed_prefix(pool: &KvPool, pages: &[Vec<u8>], cells: usize) -> Vec<u8> {
    let mut result = Vec::new();
    for p in pool.planes() {
        let width = match p {
            Plane::K | Plane::V => D,
            _ => D / 32 * 2,
        };
        let start = pool.plane_extent(0, *p).expect("checked plane").0;
        for head in 0..KV {
            for (group, page) in pages.iter().enumerate() {
                let rows = cells.min(N - group * cells);
                let begin = start + head * cells * width;
                result.extend_from_slice(&page[begin..begin + rows * width]);
            }
        }
    }
    result
}

pub(super) fn dequantize(bytes: &[u8], pool: &KvPool, code: Plane, scale: Plane) -> Vec<f64> {
    let (c, values) = pool.plane_extent(0, code).expect("code plane");
    let s = pool.plane_extent(0, scale).expect("scale plane").0;
    (0..values)
        .map(|i| {
            let at = s + (i / D * (D / 32) + (i % D) / 32) * 2;
            let a = f16(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
            f64::from(bytes[c + i] as i8) * f64::from(a)
        })
        .collect()
}

pub(super) fn reference(k: &[f64], v: &[f64], visible: usize) -> Vec<f64> {
    let rows = k.len() / (KV * D);
    assert_eq!(k.len(), v.len());
    assert!(visible > 0 && visible <= rows);
    let (q, gate) = query_and_gate();
    let mut out = vec![0.0; HEADS * D];
    for head in 0..HEADS {
        let kv = head / 8;
        let mut scores: Vec<f64> = (0..visible)
            .map(|t| {
                (0..D)
                    .map(|d| f64::from(q[head * D + d]) * k[(kv * rows + t) * D + d])
                    .sum::<f64>()
                    / 16.0
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for score in &mut scores {
            *score = (*score - max).exp();
        }
        let sum: f64 = scores.iter().sum();
        for d in 0..D {
            let mean = (0..visible)
                .map(|t| scores[t] * v[(kv * rows + t) * D + d])
                .sum::<f64>()
                / sum;
            out[head * D + d] = mean / (1.0 + (-f64::from(gate[head * 2 * D + D + d])).exp());
        }
    }
    out
}

pub(super) fn check(bytes: &[u8], expected: &[f64]) -> (f64, f64) {
    let mut max = 0.0_f64;
    let mut squared = 0.0;
    for head in 0..HEADS {
        let bound = E3
            * expected[head * D..(head + 1) * D]
                .iter()
                .map(|v| v.abs())
                .fold(1.0_f64, f64::max);
        for d in 0..D {
            let i = (head * D + d) * 4;
            let actual = f64::from(f32::from_le_bytes(
                bytes[i..i + 4].try_into().expect("float bytes"),
            ));
            let error = (actual - expected[head * D + d]).abs();
            assert!(
                actual.is_finite() && error <= ABS && error <= bound,
                "nonfinite output or frozen absolute/E3 gate: {actual} error {error}"
            );
            max = max.max(error);
            squared += error * error;
        }
    }
    (max, (squared / (HEADS * D) as f64).sqrt())
}

#[test]
fn real_chunk_append_host_reload_and_checked_attention_match_resident_and_f64() {
    if !device_present("real_chunk_append_host_reload_and_checked_attention_match_resident_and_f64")
    {
        return;
    }
    let observation = observe().expect("real host paged producer pipeline");
    let expected = packed_prefix(
        &observation.pool,
        std::slice::from_ref(&observation.reference),
        N,
    );
    let actual = packed_prefix(&observation.group, &observation.pages, C);
    assert_eq!(
        actual, expected,
        "every committed K/KScales/V/VScales byte must match resident KV8"
    );
    assert!(
        observation.isolated.iter().all(|b| *b == 0),
        "other full-attention ordinal changed"
    );
    let k = dequantize(
        &observation.reference,
        &observation.pool,
        Plane::K,
        Plane::KScales,
    );
    let v = dequantize(
        &observation.reference,
        &observation.pool,
        Plane::V,
        Plane::VScales,
    );
    let expected = reference(&k, &v, N);
    let (dense_max, _) = check(&observation.dense, &expected);
    let (max, rms) = check(&observation.paged, &expected);
    eprintln!(
        "HOST_PAGE_PIPELINE bytes_equal=true isolated=true max={max} rms={rms} dense_max={dense_max}"
    );
}
