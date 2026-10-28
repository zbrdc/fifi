//! Nonuniform token-sensitive prefix boundaries through real KV8 producer and host paging.
use super::{
    common_tests::dims,
    native_device_rig::device_present,
    paging_pipeline_tests::{
        check, dequantize, floats, inputs, plane, query_and_gate, read, reference, upload,
    },
    paging_store::{PageAttention, PageChunk, PageId, PagedKv},
    state_layout::{KvPool, Plane},
};
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceStream,
        resident::{
            attn_decode::{AttnSpan, AttnSplit},
            attn_decode_q8::{AttnDecodeQ8Call, AttnQ8Shape, attn_q8_part_bytes},
            kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape},
        },
    },
};

const ROWS: usize = 320;
const C: usize = 256;
const D: usize = 256;
const KV: usize = 2;
const H: usize = KV * 8;
const BOUNDARIES: [usize; 9] = [1, 63, 64, 65, 255, 256, 257, 319, 320];

fn produce(
    ctx: &CudaContext,
    stream: &DeviceStream,
    pool: &KvPool,
    store: &mut PagedKv,
) -> Result<DeviceBuffer, EngError> {
    let (k, v) = inputs(ROWS);
    let k = upload(ctx, stream, &floats(&k))?;
    let v = upload(ctx, stream, &floats(&v))?;
    let dense = ctx.allocate(pool.total)?;
    ctx.kv8_chunk(
        stream,
        Kv8Shape {
            rows: ROWS,
            n_kv_head: KV,
            head_dim: D,
            cells: ROWS,
            code_scale: Kv8CodeScale::StoredHalf,
        },
        Kv8Data {
            k: k.range(0, k.bytes())?,
            v: v.range(0, v.bytes())?,
            k_codes: plane(&dense, pool, Plane::K)?,
            k_scales: plane(&dense, pool, Plane::KScales)?,
            v_codes: plane(&dense, pool, Plane::V)?,
            v_scales: plane(&dense, pool, Plane::VScales)?,
        },
        0,
    )?;
    for group in 0..2 {
        let page = PageId {
            full_layer: 9,
            group,
        };
        let rows = (ROWS - group * C).min(C);
        let offset = group * C * KV * D * 4;
        store.load(ctx, stream, group, page)?;
        store.append_chunk(
            ctx,
            stream,
            group,
            PageChunk {
                page,
                position: group * C,
                rows,
                k: k.range(offset, rows * KV * D * 4)?,
                v: v.range(offset, rows * KV * D * 4)?,
            },
        )?;
        store.writeback(ctx, stream, group)?;
    }
    Ok(dense)
}

fn observe() -> Result<(), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let pool = KvPool::q8_staging_layer(&dims(), ROWS)?;
    let mut store = PagedKv::allocate(&ctx, &dims(), ROWS, 8 << 20)?;
    let codes = produce(&ctx, &stream, &pool, &mut store)?;
    let packed = read(&ctx, &stream, codes.range(0, codes.bytes())?)?;
    let k = dequantize(&packed, &pool, Plane::K, Plane::KScales);
    let v = dequantize(&packed, &pool, Plane::V, Plane::VScales);
    let (q, gate) = query_and_gate();
    let q = upload(&ctx, &stream, &floats(&q))?;
    let gate = upload(&ctx, &stream, &floats(&gate))?;
    let length = ctx.allocate(4)?;
    let part = ctx.allocate(attn_q8_part_bytes(1, KV, D, 64)?)?;
    let dense = ctx.allocate(H * D * 4)?;
    let paged = ctx.allocate(H * D * 4)?;
    let shape = AttnQ8Shape {
        m: 1,
        n_head: H,
        n_kv_head: KV,
        head_dim: D,
        cells: ROWS,
        ld: H * 2 * D,
    };
    let mut scratch = ctx.paged_q8_scratch(shape, C)?;
    let mut max = 0.0_f64;
    for visible in BOUNDARIES {
        length.write_at(&stream, 0, &(visible as i32).to_le_bytes())?;
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
                proj: Some(gate.range(0, gate.bytes())?),
                k_codes: plane(&codes, &pool, Plane::K)?,
                k_scales: plane(&codes, &pool, Plane::KScales)?,
                v_codes: plane(&codes, &pool, Plane::V)?,
                v_scales: plane(&codes, &pool, Plane::VScales)?,
                kv_len: length.range(0, 4)?,
                part: part.range(0, part.bytes())?,
                out: dense.range(0, dense.bytes())?,
            },
        )?;
        store.attend(
            &ctx,
            &stream,
            &mut scratch,
            PageAttention {
                full_layer: 9,
                visible_keys: visible,
                scale: 1.0 / 16.0,
                q: q.range(0, q.bytes())?,
                proj: Some(gate.range(0, gate.bytes())?),
                out: paged.range(0, paged.bytes())?,
            },
        )?;
        let expected = reference(&k, &v, visible);
        let adjacent = reference(&k, &v, if visible == 1 { 2 } else { visible - 1 });
        let sensitivity = expected
            .iter()
            .zip(adjacent)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            sensitivity > 2e-5,
            "boundary {visible} must detect an off-by-one prefix"
        );
        let dense_bytes = read(&ctx, &stream, dense.range(0, dense.bytes())?)?;
        let paged_bytes = read(&ctx, &stream, paged.range(0, paged.bytes())?)?;
        max = max
            .max(check(&dense_bytes, &expected).0)
            .max(check(&paged_bytes, &expected).0);
        println!("PKV_NONUNIFORM_BOUNDARY visible={visible} sensitivity={sensitivity}");
    }
    println!("PKV_NONUNIFORM_BOUNDARIES_PASS cases=9 max={max}");
    Ok(())
}

#[test]
fn nonuniform_host_pages_match_dense_and_f64_at_all_prefix_boundaries() {
    if !device_present("nonuniform_host_pages_match_dense_and_f64_at_all_prefix_boundaries") {
        return;
    }
    observe().expect("real nonuniform prefix boundary pipeline");
}
