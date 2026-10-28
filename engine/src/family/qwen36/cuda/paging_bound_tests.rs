//! Store-bound geometry/layer/logical-capacity regressions through normal native APIs.
use super::{
    common_tests::dims,
    native_device_rig::device_present,
    paging_pipeline_tests::{floats, read, upload},
    paging_store::{PageAttention, PageChunk, PageId, PagedKv},
};
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceStream,
        resident::attn_decode_q8::{AttnQ8Shape, paged::PagedQ8Scratch},
    },
};
const D: usize = 256;
const KV: usize = 2;
const H: usize = KV * 8;
const N: usize = 320;

fn shape(kv: usize, dim: usize, capacity: usize) -> AttnQ8Shape {
    AttnQ8Shape {
        m: 1,
        n_head: kv * 8,
        n_kv_head: kv,
        head_dim: dim,
        cells: capacity,
        ld: H * 2 * D,
    }
}

fn fill(
    ctx: &CudaContext,
    stream: &DeviceStream,
    store: &mut PagedKv,
    layer: usize,
    value: f32,
) -> Result<(), EngError> {
    let k = upload(ctx, stream, &floats(&vec![127.0; N * KV * D]))?;
    let values: Vec<f32> = (0..N * KV * D)
        .map(|i| {
            if (i / D).is_multiple_of(KV) {
                value
            } else {
                -value
            }
        })
        .collect();
    let v = upload(ctx, stream, &floats(&values))?;
    for group in 0..2 {
        let page = PageId {
            full_layer: layer,
            group,
        };
        let rows = if group == 0 { 256 } else { 64 };
        let offset = group * 256 * KV * D * 4;
        store.load(ctx, stream, group, page)?;
        store.append_chunk(
            ctx,
            stream,
            group,
            PageChunk {
                page,
                position: group * 256,
                rows,
                k: k.range(offset, rows * KV * D * 4)?,
                v: v.range(offset, rows * KV * D * 4)?,
            },
        )?;
        store.writeback(ctx, stream, group)?;
    }
    Ok(())
}

fn run(
    ctx: &CudaContext,
    stream: &DeviceStream,
    store: &mut PagedKv,
    scratch: &mut PagedQ8Scratch,
    io: (&DeviceBuffer, &DeviceBuffer),
    layer: usize,
    visible: usize,
) -> Result<Vec<u8>, EngError> {
    let out = ctx.allocate(H * D * 4)?;
    store.attend(
        ctx,
        stream,
        scratch,
        PageAttention {
            full_layer: layer,
            visible_keys: visible,
            scale: 1.0 / 16.0,
            q: io.0.range(0, io.0.bytes())?,
            proj: Some(io.1.range(0, io.1.bytes())?),
            out: out.range(0, out.bytes())?,
        },
    )?;
    read(ctx, stream, out.range(0, out.bytes())?)
}

fn error(bytes: &[u8], value: f32) -> (bool, f64, f64) {
    let mut max = 0.0_f64;
    let mut sum = 0.0;
    let mut pass = true;
    for (i, b) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let x = f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        let expected = if i / D / 8 == 0 {
            f64::from(value) / 2.0
        } else {
            -f64::from(value) / 2.0
        };
        let delta = (x - expected).abs();
        pass &= x.is_finite() && delta <= 1e-5 && delta <= expected.abs().max(1.0) / 131072.0;
        max = max.max(delta);
        sum += delta * delta;
    }
    (pass, max, (sum / (H * D) as f64).sqrt())
}

struct Observed {
    rejected: usize,
    unchanged: bool,
    recoveries: bool,
    boundary: bool,
    layers: bool,
    stores: bool,
    max: f64,
    rms: f64,
}

fn observe() -> Result<Observed, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let other_stream = ctx.stream()?;
    let q = upload(&ctx, &stream, &floats(&vec![0.0; H * D]))?;
    let gate = upload(&ctx, &stream, &floats(&vec![0.0; H * 2 * D]))?;
    let mut store = PagedKv::allocate(&ctx, &dims(), N, 8 << 20)?;
    fill(&ctx, &stream, &mut store, 9, 127.0)?;
    fill(&ctx, &stream, &mut store, 0, 63.5)?;
    let mut scratch = ctx.paged_q8_scratch(shape(KV, D, N), 256)?;
    eprintln!("BOUND_STORE_REAL_PRODUCER_READY");
    let first = run(&ctx, &stream, &mut store, &mut scratch, (&q, &gate), 9, N)?;
    let (first_ok, max, rms) = error(&first, 127.0);
    let mut result = Observed {
        rejected: 0,
        unchanged: true,
        recoveries: first_ok,
        boundary: true,
        layers: true,
        stores: true,
        max,
        rms,
    };
    // Every wrong scratch is independently valid; every declared input/output range is real.
    let cases = [
        (2, 256, 128, 320, 320, 9, false),
        (2, 128, 256, 320, 320, 9, false),
        (1, 256, 256, 320, 320, 9, false),
        (2, 256, 256, 512, 320, 9, false),
        (2, 256, 256, 512, 321, 9, false),
        (2, 256, 256, 512, 400, 9, false),
        (2, 256, 256, 320, 0, 9, false),
        (2, 256, 256, 320, 321, 9, false),
        (2, 256, 256, 320, usize::MAX, 9, false),
        (2, 256, 256, 320, 320, 10, false),
        (2, 256, 256, 320, 320, usize::MAX, false),
        (2, 256, 256, 320, 320, 9, true),
    ];
    for (kv, dim, cells, capacity, visible, layer, foreign_stream) in cases {
        let mut wrong = ctx.paged_q8_scratch(shape(kv, dim, capacity), cells)?;
        let sentinel = floats(&vec![73.5; H * D]);
        let out = upload(&ctx, &stream, &sentinel)?;
        let selected = if foreign_stream {
            &other_stream
        } else {
            &stream
        };
        let before = (stream.enqueued(), other_stream.enqueued());
        let refused = matches!(
            store.attend(
                &ctx,
                selected,
                &mut wrong,
                PageAttention {
                    full_layer: layer,
                    visible_keys: visible,
                    scale: 1.0 / 16.0,
                    q: q.range(0, q.bytes())?,
                    proj: Some(gate.range(0, gate.bytes())?),
                    out: out.range(0, out.bytes())?,
                }
            ),
            Err(EngError::BadRequest(_))
        );
        let no_enqueue = (stream.enqueued(), other_stream.enqueued()) == before;
        let untouched = read(&ctx, &stream, out.range(0, out.bytes())?)? == sentinel;
        result.rejected += usize::from(refused);
        result.unchanged &= no_enqueue && untouched;
        let recovered = run(&ctx, &stream, &mut store, &mut scratch, (&q, &gate), 9, N)?;
        result.recoveries &= error(&recovered, 127.0).0;
    }
    for visible in [1, 63, 64, 65, 255, 256, 257, 319, 320] {
        let bytes = run(
            &ctx,
            &stream,
            &mut store,
            &mut scratch,
            (&q, &gate),
            9,
            visible,
        )?;
        result.boundary &= error(&bytes, 127.0).0;
    }
    let layer0 = run(&ctx, &stream, &mut store, &mut scratch, (&q, &gate), 0, N)?;
    result.layers &= error(&layer0, 63.5).0;
    let mut second = PagedKv::allocate(&ctx, &dims(), N, 8 << 20)?;
    fill(&ctx, &stream, &mut second, 9, 31.75)?;
    let other = run(&ctx, &stream, &mut second, &mut scratch, (&q, &gate), 9, N)?;
    let original = run(&ctx, &stream, &mut store, &mut scratch, (&q, &gate), 9, N)?;
    result.stores &= error(&other, 31.75).0 && error(&original, 127.0).0;
    Ok(result)
}

#[test]
fn bound_store_rejects_geometry_padding_and_cross_stream_and_recovers_layers_and_stores() {
    if !device_present(
        "bound_store_rejects_geometry_padding_and_cross_stream_and_recovers_layers_and_stores",
    ) {
        return;
    }
    let result = observe().expect("real store-bound attention operation");
    assert_eq!(
        result.rejected, 12,
        "geometry/logical/layer/stream refusals"
    );
    assert!(
        result.unchanged,
        "all refusals must precede enqueues/output writes"
    );
    assert!(
        result.recoveries && result.boundary && result.layers && result.stores,
        "complete recovery, visible boundaries, full-layer and store identity"
    );
    eprintln!(
        "BOUND_STORE rejected=12 unchanged=true boundary=true layers=true stores=true max={} rms={}",
        result.max, result.rms
    );
}
