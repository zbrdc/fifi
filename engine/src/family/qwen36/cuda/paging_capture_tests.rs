//! Real capture refusal, followed by queued two-slot producer flushing and eager recovery.
use super::{
    common_tests::dims,
    native_device_rig::device_present,
    paging_pipeline_tests::{floats, read, upload},
    paging_store::{PageAttention, PageChunk, PageId, PagedKv},
};
use crate::{
    EngError,
    kernels::cuda::{CapturePlan, CudaContext, resident::attn_decode_q8::AttnQ8Shape},
};
const D: usize = 256;
const KV: usize = 2;
const H: usize = 16;

struct Observation {
    refused: usize,
    no_enqueue: bool,
    slots: bool,
    output: bool,
    recovery: bool,
    other_layer: bool,
    max: f64,
}

fn matches_values(bytes: &[u8], denominator: f64) -> (bool, f64) {
    let mut ok = true;
    let mut max = 0.0_f64;
    for (i, b) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let x = f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        let expected = if i / D / 8 == 0 {
            63.5 / denominator
        } else {
            -63.5 / denominator
        };
        let error = (x - expected).abs();
        ok &= x.is_finite() && error <= 1e-5 && error <= expected.abs().max(1.0) / 131072.0;
        max = max.max(error);
    }
    (ok, max)
}

fn observe() -> Result<Observation, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let k = upload(&ctx, &stream, &floats(&vec![127.0; KV * D]))?;
    let values: Vec<f32> = (0..KV * D)
        .map(|i| if i / D == 0 { 127.0 } else { -127.0 })
        .collect();
    let v = upload(&ctx, &stream, &floats(&values))?;
    let q = upload(&ctx, &stream, &floats(&vec![0.0; H * D]))?;
    let gate = upload(&ctx, &stream, &floats(&vec![0.0; H * 2 * D]))?;
    let warm_out = ctx.allocate(H * D * 4)?;
    let mut store = PagedKv::allocate(&ctx, &dims(), 320, 8 << 20)?;
    let a = PageId {
        full_layer: 9,
        group: 0,
    };
    let b = PageId {
        full_layer: 0,
        group: 1,
    };
    store.load(&ctx, &stream, 0, a)?;
    store.append_chunk(
        &ctx,
        &stream,
        0,
        PageChunk {
            page: a,
            position: 0,
            rows: 1,
            k: k.range(0, k.bytes())?,
            v: v.range(0, v.bytes())?,
        },
    )?;
    let mut scratch = ctx.paged_q8_scratch(
        AttnQ8Shape {
            m: 1,
            n_head: H,
            n_kv_head: KV,
            head_dim: D,
            cells: 320,
            ld: H * 2 * D,
        },
        256,
    )?;
    // Warm the real producer, partial and combine kernels before opening a capture window.
    store.attend(
        &ctx,
        &stream,
        &mut scratch,
        PageAttention {
            full_layer: 9,
            visible_keys: 1,
            scale: 1.0 / 16.0,
            q: q.range(0, q.bytes())?,
            proj: Some(gate.range(0, gate.bytes())?),
            out: warm_out.range(0, warm_out.bytes())?,
        },
    )?;
    stream.synchronize()?;
    store.load(&ctx, &stream, 0, a)?;
    store.load(&ctx, &stream, 1, b)?;
    let before0 = read(&ctx, &stream, store.resident(0)?)?;
    let before1 = read(&ctx, &stream, store.resident(1)?)?;
    let sentinel = floats(&vec![73.5; H * D]);
    let out = upload(&ctx, &stream, &sentinel)?;
    let before = stream.enqueued();
    let mut refused = 0;
    // Both operations must refuse before capturing page addresses. Always abort this normal
    // capture body while all owners are alive; no executable graph is created or replayed.
    let pins = [&k, &v, &q, &gate, &out];
    let captured=ctx.capture(&stream,CapturePlan {pins:&pins,holds:Vec::new()},|| {
        if matches!(store.append_chunk(&ctx,&stream,0,PageChunk {
            page:a,position:0,rows:1,k:k.range(0,k.bytes())?,v:v.range(0,v.bytes())?,
        }),Err(EngError::BadRequest(ref text)) if text.contains("not allowed inside a capture window")) {refused+=1;}
        if matches!(store.attend(&ctx,&stream,&mut scratch,PageAttention {
            full_layer:9,visible_keys:1,scale:1.0/16.0,q:q.range(0,q.bytes())?,
            proj:Some(gate.range(0,gate.bytes())?),out:out.range(0,out.bytes())?,
        }),Err(EngError::BadRequest(ref text)) if text.contains("not allowed inside a capture window")) {refused+=1;}
        Err::<(),_>(EngError::BadRequest("end own paged capture probe".to_owned()))
    });
    match captured {
        Err(EngError::BadRequest(text)) if text == "end own paged capture probe" => {}
        Err(error) => return Err(error),
        Ok(_) => {
            return Err(EngError::BadRequest(
                "capture probe unexpectedly made a graph".to_owned(),
            ));
        }
    }
    let no_enqueue = stream.enqueued() == before && !stream.is_capturing()?;
    let slots = read(&ctx, &stream, store.resident(0)?)? == before0
        && read(&ctx, &stream, store.resident(1)?)? == before1;
    let output = read(&ctx, &stream, out.range(0, out.bytes())?)? == sentinel;
    // No upload/readback/synchronize between these real queued writers and attend.
    store.append_chunk(
        &ctx,
        &stream,
        0,
        PageChunk {
            page: a,
            position: 0,
            rows: 1,
            k: k.range(0, k.bytes())?,
            v: v.range(0, v.bytes())?,
        },
    )?;
    store.append_chunk(
        &ctx,
        &stream,
        1,
        PageChunk {
            page: b,
            position: 256,
            rows: 1,
            k: k.range(0, k.bytes())?,
            v: v.range(0, v.bytes())?,
        },
    )?;
    store.attend(
        &ctx,
        &stream,
        &mut scratch,
        PageAttention {
            full_layer: 9,
            visible_keys: 1,
            scale: 1.0 / 16.0,
            q: q.range(0, q.bytes())?,
            proj: Some(gate.range(0, gate.bytes())?),
            out: out.range(0, out.bytes())?,
        },
    )?;
    let bytes = read(&ctx, &stream, out.range(0, out.bytes())?)?;
    let (recovery, mut max) = matches_values(&bytes, 1.0);
    // Only the other layer's global row256 was written; its256 earlier rows are valid zeros.
    store.attend(
        &ctx,
        &stream,
        &mut scratch,
        PageAttention {
            full_layer: 0,
            visible_keys: 257,
            scale: 1.0 / 16.0,
            q: q.range(0, q.bytes())?,
            proj: Some(gate.range(0, gate.bytes())?),
            out: out.range(0, out.bytes())?,
        },
    )?;
    let bytes = read(&ctx, &stream, out.range(0, out.bytes())?)?;
    let (other_layer, error) = matches_values(&bytes, 257.0);
    max = max.max(error);
    Ok(Observation {
        refused,
        no_enqueue,
        slots,
        output,
        recovery,
        other_layer,
        max,
    })
}

#[test]
fn warmed_paged_operations_refuse_capture_and_flush_queued_two_slot_writers() {
    if !device_present("warmed_paged_operations_refuse_capture_and_flush_queued_two_slot_writers") {
        return;
    }
    let x = observe().expect("real capture-refusal and queued-flush probe");
    assert_eq!(x.refused, 2);
    assert!(
        x.no_enqueue && x.slots && x.output,
        "capture refusal changed queued work/residency/output"
    );
    assert!(
        x.recovery && x.other_layer,
        "eager recovery/automatic dirty-slot flush lost layer data"
    );
    eprintln!(
        "PAGED_CAPTURE refused=2 no_enqueue=true slots=true output=true recovery=true other_layer=true max={}",
        x.max
    );
}

#[test]
fn paged_allocation_refuses_capture_before_host_quota_admission() {
    if !device_present("paged_allocation_refuses_capture_before_host_quota_admission") {
        return;
    }
    let ctx = CudaContext::new().expect("capture admission context");
    let stream = ctx.stream().expect("capture admission stream");
    let pin = ctx.allocate(16).expect("live capture pin");
    let pins = [&pin];
    let before = (ctx.live_bytes(), stream.enqueued());
    let mut observed = None;
    let capture = ctx.capture(
        &stream,
        CapturePlan {
            pins: &pins,
            holds: Vec::new(),
        },
        || {
            assert!(stream.is_capturing()?);
            observed = Some(match PagedKv::allocate(&ctx, &dims(), 320, 0) {
                Err(error) => error,
                Ok(_) => panic!("zero host quota cannot admit a store"),
            });
            Err::<(), _>(EngError::BadRequest(
                "abort allocation admission probe".to_owned(),
            ))
        },
    );
    assert!(matches!(capture, Err(EngError::BadRequest(ref text))
                     if text == "abort allocation admission probe"));
    assert!(!stream.is_capturing().expect("capture ended"));
    println!("PKV_CAPTURE_ADMISSION_READY actual_capture=true quota=0 observed={observed:?}");
    assert!(
        matches!(observed, Some(EngError::BadRequest(ref text))
                     if text.contains("not allowed inside a capture window")),
        "capture refusal must precede host quota admission"
    );
    assert_eq!((ctx.live_bytes(), stream.enqueued()), before);
    assert!(
        matches!(
            PagedKv::allocate(&ctx, &dims(), 320, 0),
            Err(EngError::MemoryPlan(_))
        ),
        "outside capture, host quota is still enforced"
    );
    let store = PagedKv::allocate(&ctx, &dims(), 320, 8 << 20).expect("eager recovery");
    drop(store);
    assert_eq!(ctx.live_bytes(), before.0);
    println!("PKV_CAPTURE_ADMISSION_PASS eager_recovery=true");
}
