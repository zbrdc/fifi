//! Normal-API page partial/combine against independent dense input and f64 closed form.
use super::{
    AttnDecodeQ8Call, AttnQ8CombineCall, AttnQ8PartialCall, AttnQ8Shape, AttnSpan, AttnSplit,
    attn_q8_part_bytes,
};
use crate::EngError;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, device_present};

const D: usize = 256;
const KV: usize = 2;
const G: usize = 8;
const C: usize = 256;
const N: usize = 257;
const P: usize = 64;
const ONE: u16 = 0x3c00;
const POISON: u16 = 0x7e00;
const ABS: f64 = 1e-5;
const E3: f64 = 1.0 / 131_072.0;

struct Rows {
    cells: usize,
    k: Vec<i8>,
    v: Vec<i8>,
    ks: Vec<u16>,
    vs: Vec<u16>,
}

fn base(head: usize, dim: usize) -> i8 {
    if head == 0 {
        1 + (dim % 7) as i8
    } else {
        2 - (dim % 5) as i8
    }
}

fn tail(head: usize, dim: usize) -> i8 {
    if head == 0 {
        3 - (dim % 3) as i8
    } else {
        -4 + (dim % 9) as i8
    }
}

fn tail_key(head: usize) -> i8 {
    if head == 0 { 1 } else { -2 }
}

fn dense() -> Rows {
    let mut rows = Rows {
        cells: N,
        k: vec![0; KV * N * D],
        v: vec![0; KV * N * D],
        ks: vec![ONE; KV * N * (D / 32)],
        vs: vec![ONE; KV * N * (D / 32)],
    };
    for head in 0..KV {
        for token in 0..N {
            for dim in 0..D {
                let offset = (head * N + token) * D + dim;
                if token == N - 1 && dim == 0 {
                    rows.k[offset] = tail_key(head);
                }
                rows.v[offset] = if token == N - 1 {
                    tail(head, dim)
                } else {
                    base(head, dim)
                };
            }
        }
    }
    rows
}

// Built in group/head/local-row order, not gathered from dense().
fn page(group: usize) -> Rows {
    let visible = if group == 0 { C } else { 1 };
    let mut rows = Rows {
        cells: C,
        k: vec![127; KV * C * D],
        v: vec![-127; KV * C * D],
        ks: vec![POISON; KV * C * (D / 32)],
        vs: vec![POISON; KV * C * (D / 32)],
    };
    for head in 0..KV {
        for local in 0..visible {
            for dim in 0..D {
                let offset = head * C * D + local * D + dim;
                rows.k[offset] = if group == 1 && dim == 0 {
                    tail_key(head)
                } else {
                    0
                };
                rows.v[offset] = if group == 1 {
                    tail(head, dim)
                } else {
                    base(head, dim)
                };
            }
            for quant_group in 0..D / 32 {
                let offset = (head * C + local) * (D / 32) + quant_group;
                rows.ks[offset] = ONE;
                rows.vs[offset] = ONE;
            }
        }
    }
    rows
}

fn shape(cells: usize) -> AttnQ8Shape {
    AttnQ8Shape {
        m: 1,
        n_head: KV * G,
        n_kv_head: KV,
        head_dim: D,
        cells,
        ld: 0,
    }
}

fn closed_form(head: usize, dim: usize) -> f64 {
    let kvh = head / G;
    let score = (head + 1) as f64 / 8.0 * f64::from(tail_key(kvh));
    (256.0 * f64::from(base(kvh, dim)) + score.exp() * f64::from(tail(kvh, dim)))
        / (256.0 + score.exp())
}

fn fixture_error(message: &str) -> EngError {
    EngError::BadRequest(format!("paged attention fixture: {message}"))
}

struct ErrorStats {
    max: f64,
    rms: f64,
}

// Apply the pre-frozen absolute bound AND the unchanged per-head E3 bound.
fn metrics(actual: &[f32], expected: &[f64]) -> Result<ErrorStats, EngError> {
    if actual.len() != KV * G * D || expected.len() != actual.len() {
        return Err(fixture_error("output length"));
    }
    let mut max = 0.0_f64;
    let mut squared = 0.0;
    for head in 0..KV * G {
        let start = head * D;
        let reference = &expected[start..start + D];
        let bound = E3 * reference.iter().map(|v| v.abs()).fold(1.0_f64, f64::max);
        for dim in 0..D {
            let got = f64::from(actual[start + dim]);
            let error = (got - reference[dim]).abs();
            if !got.is_finite() || error > ABS || error > bound {
                return Err(fixture_error("nonfinite output or frozen absolute/E3 gate"));
            }
            max = max.max(error);
            squared += error * error;
        }
    }
    Ok(ErrorStats {
        max,
        rms: (squared / actual.len() as f64).sqrt(),
    })
}

struct GpuRows {
    k: DeviceBuffer,
    v: DeviceBuffer,
    ks: DeviceBuffer,
    vs: DeviceBuffer,
}

impl GpuRows {
    fn upload(ctx: &CudaContext, rows: &Rows) -> Result<Self, EngError> {
        Ok(Self {
            k: ctx.upload(&rows.k.iter().map(|v| *v as u8).collect::<Vec<_>>())?,
            v: ctx.upload(&rows.v.iter().map(|v| *v as u8).collect::<Vec<_>>())?,
            ks: ctx.upload(
                &rows
                    .ks
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
            vs: ctx.upload(
                &rows
                    .vs
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
        })
    }
}

fn upload_f32(ctx: &CudaContext, values: &[f32]) -> Result<DeviceBuffer, EngError> {
    ctx.upload(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn download(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>, EngError> {
    let mut bytes = vec![0; KV * G * D * 4];
    ctx.download(buffer, &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn run_dense(
    ctx: &CudaContext,
    stream: &DeviceStream,
    q: &DeviceBuffer,
    rows: &Rows,
) -> Result<Vec<f32>, EngError> {
    let gpu = GpuRows::upload(ctx, rows)?;
    let part = ctx.upload(&vec![0xff; attn_q8_part_bytes(1, KV, D, P)?])?;
    let out = ctx.upload(&vec![0xff; KV * G * D * 4])?;
    let len = ctx.upload(&(N as i32).to_le_bytes())?;
    ctx.attn_decode_q8(
        stream,
        AttnDecodeQ8Call {
            shape: shape(rows.cells),
            span: AttnSpan::FULL,
            split: AttnSplit {
                max_splits: P,
                min_keys: C,
            },
            scale: 1.0,
            q: q.range(0, q.bytes())?,
            proj: None,
            k_codes: gpu.k.range(0, gpu.k.bytes())?,
            k_scales: gpu.ks.range(0, gpu.ks.bytes())?,
            v_codes: gpu.v.range(0, gpu.v.bytes())?,
            v_scales: gpu.vs.range(0, gpu.vs.bytes())?,
            kv_len: len.range(0, 4)?,
            part: part.range(0, part.bytes())?,
            out: out.range(0, out.bytes())?,
        },
    )?;
    stream.synchronize()?;
    download(ctx, &out)
}

fn run_pages(
    ctx: &CudaContext,
    stream: &DeviceStream,
    q: &DeviceBuffer,
    pages: &[Rows; 2],
) -> Result<Vec<f32>, EngError> {
    let part = ctx.upload(&vec![0xff; attn_q8_part_bytes(1, KV, D, P)?])?;
    let out = ctx.upload(&vec![0xff; KV * G * D * 4])?;
    for (group, rows) in pages.iter().enumerate() {
        let gpu = GpuRows::upload(ctx, rows)?;
        ctx.attn_decode_q8_partial(
            stream,
            AttnQ8PartialCall {
                shape: shape(rows.cells),
                visible_keys: if group == 0 { C } else { 1 },
                group,
                scale: 1.0,
                q: q.range(0, q.bytes())?,
                k_codes: gpu.k.range(0, gpu.k.bytes())?,
                k_scales: gpu.ks.range(0, gpu.ks.bytes())?,
                v_codes: gpu.v.range(0, gpu.v.bytes())?,
                v_scales: gpu.vs.range(0, gpu.vs.bytes())?,
                part: part.range(0, part.bytes())?,
            },
        )?;
        // Local owners must outlive their queued read before the next page replaces them.
        stream.synchronize()?;
    }
    ctx.attn_decode_q8_combine(
        stream,
        AttnQ8CombineCall {
            shape: shape(C * 2),
            page_cells: C,
            visible_keys: N,
            part: part.range(0, part.bytes())?,
            proj: None,
            out: out.range(0, out.bytes())?,
        },
    )?;
    stream.synchronize()?;
    download(ctx, &out)
}

fn observations() -> Result<(ErrorStats, ErrorStats), EngError> {
    let dense_rows = dense();
    let pages = [page(0), page(1)];
    let expected: Vec<f64> = (0..KV * G)
        .flat_map(|head| (0..D).map(move |dim| closed_form(head, dim)))
        .collect();
    let mut queries = vec![0.0; KV * G * D];
    for head in 0..KV * G {
        queries[head * D] = (head + 1) as f32 / 8.0;
    }
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let q = upload_f32(&ctx, &queries)?;
    let reference = run_dense(&ctx, &stream, &q, &dense_rows)?;
    let dense_stats = metrics(&reference, &expected)?;
    eprintln!(
        "PAGED_DENSE_CONTROL_PASS max={} rms={}",
        dense_stats.max, dense_stats.rms
    );
    eprintln!("PAGED_PARTIAL_REAL_DEVICE_READY");
    let paged = run_pages(&ctx, &stream, &q, &pages)?;
    let paged_stats = metrics(&paged, &expected)?;
    let resident_reference: Vec<f64> = reference.iter().map(|v| f64::from(*v)).collect();
    let pair_stats = metrics(&paged, &resident_reference)?;
    eprintln!(
        "PAGED_ATTENTION_ERRORS paged_max={} paged_rms={} pair_max={} pair_rms={}",
        paged_stats.max, paged_stats.rms, pair_stats.max, pair_stats.rms
    );
    Ok((dense_stats, paged_stats))
}

#[test]
fn paged_partials_and_combine_match_independent_dense_and_f64() {
    if !device_present("paged_partials_and_combine_match_independent_dense_and_f64") {
        return;
    }
    let _stats = observations().expect("native paged attention partial/combine");
}

#[test]
fn paged_fixture_committed_rows_match_independent_dense() {
    let full = dense();
    for group in 0..2 {
        let p = page(group);
        let visible = if group == 0 { C } else { 1 };
        for head in 0..KV {
            for local in 0..visible {
                let f = (head * N + group * C + local) * D;
                let g = (head * C + local) * D;
                assert_eq!(&full.k[f..f + D], &p.k[g..g + D]);
                assert_eq!(&full.v[f..f + D], &p.v[g..g + D]);
                let fs = (head * N + group * C + local) * (D / 32);
                let gs = (head * C + local) * (D / 32);
                assert_eq!(&full.ks[fs..fs + D / 32], &p.ks[gs..gs + D / 32]);
                assert_eq!(&full.vs[fs..fs + D / 32], &p.vs[gs..gs + D / 32]);
            }
        }
        if group == 1 {
            assert_eq!(p.ks[D / 32], POISON);
        }
    }
}

// Identical split boundaries isolate the paged TU from resident split-plan differences.
fn identity_rows(d: usize, cells: usize, first: usize) -> Rows {
    let mut rows = Rows {
        cells,
        k: Vec::new(),
        v: Vec::new(),
        ks: Vec::new(),
        vs: Vec::new(),
    };
    for head in 0..KV {
        for local in 0..cells {
            let token = first + local;
            for dim in 0..d {
                rows.k
                    .push(((token * 7 + head * 11 + dim * 3) % 17) as i8 - 8);
                rows.v
                    .push(((token * 13 + head * 3 + dim * 5) % 23) as i8 - 11);
            }
            for quant in 0..d / 32 {
                rows.ks
                    .push([0x3400, 0x3800, ONE][(token + head + quant) % 3]);
                rows.vs
                    .push([0x3800, ONE, 0x4000][(token + head + quant) % 3]);
            }
        }
    }
    rows
}

fn identity_case(d: usize, keys: usize, gated: bool) -> Result<(), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let heads = KV * G;
    let shape = AttnQ8Shape {
        m: 1,
        n_head: heads,
        n_kv_head: KV,
        head_dim: d,
        cells: keys,
        ld: heads * 2 * d,
    };
    let query: Vec<f32> = (0..heads * d)
        .map(|i| (i % 19) as f32 / 64.0 - 0.125)
        .collect();
    let q = upload_f32(&ctx, &query)?;
    let mut gate = vec![f32::NAN; heads * 2 * d];
    for head in 0..heads {
        for dim in 0..d {
            gate[head * 2 * d + d + dim] = ((head + dim) % 13) as f32 / 8.0 - 0.75;
        }
    }
    let projection = upload_f32(&ctx, &gate)?;
    let proj = if gated {
        Some(projection.range(0, projection.bytes())?)
    } else {
        None
    };
    let dense = GpuRows::upload(&ctx, &identity_rows(d, keys, 0))?;
    let scratch_bytes = attn_q8_part_bytes(1, KV, d, P)?;
    let resident_part = ctx.upload(&vec![0xff; scratch_bytes])?;
    let paged_part = ctx.upload(&vec![0xff; scratch_bytes])?;
    let resident_out = ctx.allocate(heads * d * 4)?;
    let paged_out = ctx.allocate(heads * d * 4)?;
    let len = ctx.upload(&(keys as i32).to_le_bytes())?;
    ctx.attn_decode_q8(
        &stream,
        AttnDecodeQ8Call {
            shape,
            span: AttnSpan::FULL,
            split: AttnSplit {
                max_splits: P,
                min_keys: C,
            },
            scale: 0.25,
            q: q.range(0, q.bytes())?,
            proj,
            k_codes: dense.k.range(0, dense.k.bytes())?,
            k_scales: dense.ks.range(0, dense.ks.bytes())?,
            v_codes: dense.v.range(0, dense.v.bytes())?,
            v_scales: dense.vs.range(0, dense.vs.bytes())?,
            kv_len: len.range(0, 4)?,
            part: resident_part.range(0, scratch_bytes)?,
            out: resident_out.range(0, resident_out.bytes())?,
        },
    )?;
    let groups = keys.div_ceil(C);
    for group in 0..groups {
        let page = GpuRows::upload(&ctx, &identity_rows(d, C, group * C))?;
        ctx.attn_decode_q8_partial(
            &stream,
            AttnQ8PartialCall {
                shape: AttnQ8Shape { cells: C, ..shape },
                visible_keys: (keys - group * C).min(C),
                group,
                scale: 0.25,
                q: q.range(0, q.bytes())?,
                k_codes: page.k.range(0, page.k.bytes())?,
                k_scales: page.ks.range(0, page.ks.bytes())?,
                v_codes: page.v.range(0, page.v.bytes())?,
                v_scales: page.vs.range(0, page.vs.bytes())?,
                part: paged_part.range(0, scratch_bytes)?,
            },
        )?;
        stream.synchronize()?;
    }
    ctx.attn_decode_q8_combine(
        &stream,
        AttnQ8CombineCall {
            shape: AttnQ8Shape {
                cells: C * P,
                ..shape
            },
            page_cells: C,
            visible_keys: keys,
            part: paged_part.range(0, scratch_bytes)?,
            proj,
            out: paged_out.range(0, paged_out.bytes())?,
        },
    )?;
    stream.synchronize()?;
    let mut resident = vec![0; scratch_bytes];
    let mut paged = vec![0; scratch_bytes];
    ctx.download(&resident_part, &mut resident)?;
    ctx.download(&paged_part, &mut paged)?;
    for head in 0..KV {
        for group in 0..groups {
            let cell = head * P + group;
            // The native partial buffer stores every statistic before any accumulator.
            for (start, bytes) in [
                (cell * G * 2 * 4, G * 2 * 4),
                ((KV * P * G * 2 + cell * G * d) * 4, G * d * 4),
            ] {
                assert_eq!(
                    &paged[start..start + bytes],
                    &resident[start..start + bytes],
                    "partial D{d} keys{keys} gate{gated} head{head} group{group} offset{start}"
                );
            }
        }
    }
    resident.resize(resident_out.bytes(), 0);
    paged.resize(paged_out.bytes(), 0);
    ctx.download(&resident_out, &mut resident)?;
    ctx.download(&paged_out, &mut paged)?;
    assert!(
        resident
            .as_chunks::<4>()
            .0
            .iter()
            .all(|b| f32::from_le_bytes(*b).is_finite())
    );
    assert!(
        resident
            .as_chunks::<4>()
            .0
            .iter()
            .any(|b| f32::from_le_bytes(*b) != 0.0)
    );
    assert_eq!(paged, resident, "output D{d} keys{keys} gate{gated}");
    eprintln!(
        "PAGED_IDENTITY D={d} keys={keys} gated={gated} groups={groups} partials_equal=true output_equal=true"
    );
    Ok(())
}

#[test]
fn identity_pages_are_bit_exact_for_d128_and_d256() {
    if !device_present("identity_pages_are_bit_exact_for_d128_and_d256") {
        return;
    }
    for d in [128, 256] {
        for keys in [1023, 1024, 16383, 16384] {
            for gated in [false, true] {
                identity_case(d, keys, gated).expect("paged/resident identity boundaries");
            }
        }
    }
}
