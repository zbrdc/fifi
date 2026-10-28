//! Prime valid finite scratch, then expose a stale generation without any invalid GPU address.
use super::{
    AttnQ8Shape, CudaContext, EngError,
    paged::{PagedQ8Input, PagedQ8Page},
};
use crate::kernels::cuda::{DeviceBuffer, PinnedBuf, device_present};

const D: usize = 128;
const HEADS: usize = 8;
const C: usize = 64;
const N: usize = 65;
const CODES: usize = C * D;
const SCALES: usize = C * (D / 32) * 2;
const OUT: usize = HEADS * D * 4;

fn page_bytes(last: bool) -> Vec<u8> {
    let mut bytes = vec![0; 2 * (CODES + SCALES)];
    let v = CODES + SCALES;
    let vs = v + CODES;
    for row in 0..C {
        let live = !last || row == 0;
        for dim in 0..D {
            bytes[row * D + dim] = if last && dim == 0 { 1 } else { 0 };
            bytes[v + row * D + dim] = if last { 9 } else { 1 };
        }
        let bits: u16 = if live { 0x3c00 } else { 0x7e00 };
        for group in 0..D / 32 {
            let i = (row * (D / 32) + group) * 2;
            bytes[CODES + i..CODES + i + 2].copy_from_slice(&bits.to_le_bytes());
            bytes[vs + i..vs + i + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    bytes
}

fn page(buffer: &DeviceBuffer, group: usize) -> Result<PagedQ8Page<'_>, EngError> {
    Ok(PagedQ8Page {
        group,
        k_codes: buffer.range(0, CODES)?,
        k_scales: buffer.range(CODES, SCALES)?,
        v_codes: buffer.range(CODES + SCALES, CODES)?,
        v_scales: buffer.range(2 * CODES + SCALES, SCALES)?,
    })
}

fn query(value: f32) -> Vec<u8> {
    let mut q = vec![0.0_f32; HEADS * D];
    for head in 0..HEADS {
        q[head * D] = value;
    }
    q.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<u8>, EngError> {
    let mut bytes = vec![0; OUT];
    ctx.download(buffer, &mut bytes)?;
    Ok(bytes)
}

fn bound(score: f64) -> f64 {
    let reference = (64.0 + 9.0 * score.exp()) / (64.0 + score.exp());
    1e-5_f64.min(reference.abs().max(1.0) / 131_072.0)
}

fn error(bytes: &[u8], score: f64) -> f64 {
    let expected = (64.0 + 9.0 * score.exp()) / (64.0 + score.exp());
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| {
            let value = f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            if value.is_finite() {
                (value - expected).abs()
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0, f64::max)
}

struct Observed {
    abandoned: bool,
    refused: bool,
    untouched: bool,
    prime_error: f64,
    recovery_error: f64,
}

fn observations() -> Result<Observed, EngError> {
    let p0 = page_bytes(false);
    let p1 = page_bytes(true);
    let a = query(0.25);
    let b = query(2.0);
    let sentinel = vec![0xff; OUT];
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let q_a = ctx.upload(&a)?;
    let q_b = ctx.upload(&b)?;
    let gpu0 = ctx.upload(&p0)?;
    let gpu1 = ctx.upload(&p1)?;
    let prime = ctx.upload(&sentinel)?;
    let output = ctx.upload(&sentinel)?;
    let shape = AttnQ8Shape {
        m: 1,
        n_head: HEADS,
        n_kv_head: 1,
        head_dim: D,
        cells: 2 * C,
        ld: 0,
    };
    let mut scratch = ctx.paged_q8_scratch(shape, C)?;
    let mut pass = scratch.begin(
        &ctx,
        &stream,
        PagedQ8Input {
            q: q_a.range(0, a.len())?,
            visible_keys: N,
            scale: 1.0,
        },
    )?;
    pass.push(page(&gpu0, 0)?)?;
    pass.push(page(&gpu1, 1)?)?;
    pass.finish(None, prime.range(0, OUT)?)?;
    stream.synchronize()?;
    let prime_error = error(&read(&ctx, &prime)?, 0.25);
    if prime_error > bound(0.25) {
        return Err(EngError::BadRequest(
            "composition prime fixture failed".to_owned(),
        ));
    }
    eprintln!("COMPOSITION_PRIME_PASS max={prime_error}");
    {
        let mut abandoned = scratch.begin(
            &ctx,
            &stream,
            PagedQ8Input {
                q: q_b.range(0, b.len())?,
                visible_keys: N,
                scale: 1.0,
            },
        )?;
        abandoned.push(page(&gpu0, 0)?)?;
    }
    stream.synchronize()?;
    let abandoned = read(&ctx, &output)? == sentinel;
    let mut pass = scratch.begin(
        &ctx,
        &stream,
        PagedQ8Input {
            q: q_b.range(0, b.len())?,
            visible_keys: N,
            scale: 1.0,
        },
    )?;
    pass.push(page(&gpu0, 0)?)?;
    eprintln!("COMPOSITION_STALE_REAL_DEVICE_READY");
    let refused = matches!(
        pass.finish(None, output.range(0, OUT)?),
        Err(EngError::BadRequest(_))
    );
    stream.synchronize()?;
    let untouched = read(&ctx, &output)? == sentinel;
    let mut pass = scratch.begin(
        &ctx,
        &stream,
        PagedQ8Input {
            q: q_b.range(0, b.len())?,
            visible_keys: N,
            scale: 1.0,
        },
    )?;
    pass.push(page(&gpu0, 0)?)?;
    pass.push(page(&gpu1, 1)?)?;
    pass.finish(None, prime.range(0, OUT)?)?;
    stream.synchronize()?;
    let recovery_error = error(&read(&ctx, &prime)?, 2.0);
    eprintln!(
        "COMPOSITION_OBSERVATION refused={refused} untouched={untouched} recovery_max={recovery_error}"
    );
    println!("COMPOSITION_ABANDONED_PASS no_publish={abandoned} recovery_max={recovery_error}");
    Ok(Observed {
        abandoned,
        refused,
        untouched,
        prime_error,
        recovery_error,
    })
}

#[test]
fn checked_pass_refuses_missing_and_stale_groups_without_output_write() {
    if !device_present("checked_pass_refuses_missing_and_stale_groups_without_output_write") {
        return;
    }
    let got = observations().expect("normal checked composition observations");
    assert!(
        got.refused,
        "a fresh query must not combine a stale group from the previous pass"
    );
    assert!(
        got.untouched,
        "a refused incomplete pass must not overwrite output"
    );
    assert!(got.abandoned, "abandonment must not publish output");
    assert!(got.prime_error <= bound(0.25));
    assert!(got.recovery_error <= bound(2.0));
}

fn pinned_source(ctx: &CudaContext, bytes: &[u8]) -> Result<PinnedBuf, EngError> {
    let mut pinned = ctx.pinned_buf(bytes.len())?;
    pinned.write(0, bytes)?;
    Ok(pinned)
}

fn joined_producer() -> Result<f64, EngError> {
    let ctx = CudaContext::new()?;
    let producer = ctx.stream()?;
    let consumer = ctx.stream()?;
    let ready = ctx.event()?;
    let done = ctx.event()?;
    let query_bytes = query(2.0);
    let first = page_bytes(false);
    let last = page_bytes(true);
    let q = ctx.allocate(query_bytes.len())?;
    let gpu0 = ctx.allocate(first.len())?;
    let gpu1 = ctx.allocate(last.len())?;
    let out = ctx.allocate(OUT)?;
    let mut q_source = pinned_source(&ctx, &query_bytes)?;
    let mut first_source = pinned_source(&ctx, &first)?;
    let mut last_source = pinned_source(&ctx, &last)?;
    let mut result = ctx.pinned_buf(OUT)?;
    let shape = AttnQ8Shape {
        m: 1,
        n_head: HEADS,
        n_kv_head: 1,
        head_dim: D,
        cells: 2 * C,
        ld: 0,
    };
    let mut scratch = ctx.paged_q8_scratch(shape, C)?;
    // No host fence separates producer submissions from the consumer's event join.
    ctx.copy_from_pinned(&producer, &mut q_source, 0, q.range(0, q.bytes())?)?;
    ctx.copy_from_pinned(
        &producer,
        &mut first_source,
        0,
        gpu0.range(0, gpu0.bytes())?,
    )?;
    ctx.copy_from_pinned(&producer, &mut last_source, 0, gpu1.range(0, gpu1.bytes())?)?;
    producer.record(&ready)?;
    assert!(q_source.is_pending() && first_source.is_pending() && last_source.is_pending());
    let before = consumer.enqueued();
    consumer.wait(&ready)?;
    assert_eq!(
        consumer.enqueued(),
        before + 1,
        "the consumer must enqueue the join"
    );
    let mut pass = scratch.begin(
        &ctx,
        &consumer,
        PagedQ8Input {
            q: q.range(0, q.bytes())?,
            visible_keys: N,
            scale: 1.0,
        },
    )?;
    pass.push(page(&gpu0, 0)?)?;
    pass.push(page(&gpu1, 1)?)?;
    pass.finish(None, out.range(0, OUT)?)?;
    ctx.copy_to_pinned(&consumer, out.range(0, OUT)?, &mut result, 0)?;
    consumer.record(&done)?;
    result.wait(&done)?;
    let mut bytes = vec![0; OUT];
    result.read(0, &mut bytes)?;
    assert!(q_source.is_pending() && first_source.is_pending() && last_source.is_pending());
    q_source.wait(&ready)?;
    first_source.wait(&ready)?;
    last_source.wait(&ready)?;
    let max = error(&bytes, 2.0);
    println!(
        "CHECKED_PRODUCER_JOIN_PASS pending=true join_enqueued=true host_wait_before_consumer=false max={max}"
    );
    Ok(max)
}

#[test]
fn checked_consumer_joins_async_producer_without_a_host_fence() {
    if !device_present("checked_consumer_joins_async_producer_without_a_host_fence") {
        return;
    }
    let max = joined_producer().expect("real producer/event/consumer composition");
    assert!(
        max <= bound(2.0),
        "joined consumer must meet the frozen f64 gate"
    );
}
