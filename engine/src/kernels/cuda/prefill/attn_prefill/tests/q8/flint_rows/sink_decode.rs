use super::*;
use crate::family::flint::{cuda_kv::LayerKvAllocation, kv_plan::KvPlan};

#[test]
fn sink_decode_correctness_entry_accepts_prefill_call_type() {
    let _: fn(&CudaContext, &DeviceStream, AttnPrefillCall<'_>) -> Result<(), EngError> =
        |ctx, stream, call| ctx.attn_decode_sinks_via_prefill(stream, call);
}

// Distinguish absolute generations; the inherited payload alone repeats every 256 rows.
fn epoch_plane(
    shape: AttnPrefillShape,
    role: usize,
    origin: usize,
    storage: AttnKvStorage,
) -> (Plane, Vec<f64>) {
    let (mut physical, mut logical) = staged_plane(shape, role, origin);
    let scalar = physical.layout.scales.expect("fixture scales");
    let end = shape.p0 as usize + 1;
    for row in 0..shape.cache_rows {
        let absolute = origin + row;
        let target = absolute % shape.cache_rows;
        let factor = (1 + absolute / 256) as f32;
        for h in 0..KV_HEADS {
            for g in 0..DIM / 32 {
                let scale = if absolute < end {
                    (1 + (row + h + g + role) % 4) as f32 / 128.0 * factor
                } else {
                    f32::NAN
                };
                let offset = ORIGINS[1]
                    + target * scalar.strides.time_stride_bytes
                    + h * scalar.strides.head_stride_bytes
                    + g * scalar.group_stride_bytes;
                put_scalar(&mut physical.bytes[1], offset, Dtype::F16, scale);
            }
            for d in 0..DIM {
                let source = (row * KV_HEADS + h) * DIM + d;
                let value = if absolute < end {
                    logical[source] * f64::from(factor)
                } else {
                    f64::NAN
                };
                let value = if storage == AttnKvStorage::F16 {
                    f64::from(f16_to_f32(f16_bits(value as f32)))
                } else {
                    value
                };
                logical[source] = value;
                physical.values[(target * KV_HEADS + h) * DIM + d] = value;
            }
        }
    }
    (physical, logical)
}

struct Observation {
    actual: Vec<f32>,
    later: Vec<f32>,
    without_sink: Vec<f32>,
    reference: Vec<f64>,
    status: i32,
    guards: bool,
    refusals: bool,
    refusal_results: Vec<String>,
    distinct_epochs: bool,
}

// Keep the device phase Result-only; all assertions run after every owner is dropped.
fn observe(
    layer: usize,
    shape: AttnPrefillShape,
    span: CpfAttnSpan,
    storage: AttnKvStorage,
) -> Result<Observation, EngError> {
    let origin = if span.ring > 0 {
        (shape.p0 as usize).saturating_sub(WINDOW - 1)
    } else {
        0
    };
    let (k, logical_k) = epoch_plane(shape, 0, origin, storage);
    let (v, logical_v) = epoch_plane(shape, 1, origin, storage);
    let width = KV_HEADS * DIM;
    let distinct_epochs = logical_v[..width] != logical_v[256 * width..257 * width];
    let input = Data {
        q: (0..HEADS * DIM)
            .map(|i| ((i % 31) as f32 - 15.0) / 512.0)
            .collect(),
        k: k.values.iter().map(|&x| f16_bits(x as f32)).collect(),
        v: v.values.iter().map(|&x| f16_bits(x as f32)).collect(),
        proj: Vec::new(),
    };
    let sinks: Vec<f32> = (0..HEADS)
        .map(|h| [-4.0, 0.5, 12.0][h % 3] + (layer % 7) as f32 / 8.0 + h as f32 / 64.0)
        .collect();
    let q: Vec<f64> = input.q.iter().copied().map(f64::from).collect();
    let sink_ref: Vec<f64> = sinks.iter().copied().map(f64::from).collect();
    let reference = sdpa_f64(
        &q,
        &logical_k,
        &logical_v,
        1,
        HEADS,
        shape.cache_rows,
        DIM,
        HEADS / KV_HEADS,
        0.125,
        (shape.p0 as usize - origin) as u32,
        0,
        (span.window > 0).then_some(span.window as u32),
        Some(&sink_ref),
    );
    let split = CpfAttnSplit {
        max_splits: 3,
        min_keys: 32,
    };
    let mut fx = Fx::with_data(shape, split, false, true, &input)?;
    fx.span = span;
    fx.scale = 0.125;
    fx.sink_buf = Some(fx.ctx.upload(&le(&sinks))?);
    let buffers = if storage == AttnKvStorage::F16 {
        None
    } else {
        Some([k.upload(&fx.ctx)?, v.upload(&fx.ctx)?])
    };
    let views = match &buffers {
        Some([kb, vb]) => Some([k.view(kb)?, v.view(vb)?]),
        None => None,
    };
    let call = || -> Result<AttnPrefillCall<'_>, EngError> {
        let mut call = fx.call()?;
        call.storage = storage;
        if let Some(views) = views {
            call.k_cache = views[0].payload;
            call.v_cache = views[1].payload;
            call.kv_views = Some(views);
        }
        Ok(call)
    };
    let inputs = [fx.bytes(Q), fx.bytes(KC), fx.bytes(VC), fx.bytes(PROJ)];
    fx.ctx.attn_decode_sinks_via_prefill(&fx.stream, call()?)?;
    fx.stream.synchronize()?;
    let actual = fx.out_values();
    fx.reset_outputs()?;
    let mut omitted = call()?;
    omitted.sinks = None;
    fx.ctx.attn_prefill(&fx.stream, omitted)?;
    fx.stream.synchronize()?;
    let without_sink = fx.out_values();
    let before = [fx.bytes(PART), fx.bytes(OUT), fx.bytes(STATUS)];
    let mut refusals = true;
    let mut refusal_results = Vec::new();
    for invalid in 0..11 {
        let mut bad = call()?;
        match invalid {
            0 => bad.shape.m = 0,
            1 => bad.shape.m = 2,
            2 => bad.sinks = None,
            3 => {
                bad.sinks = fx
                    .sink_buf
                    .as_ref()
                    .map(|b| b.range(0, KV_HEADS * 4))
                    .transpose()?
            }
            4 => bad.sinks = Some(fx.bufs[OUT].range(0, HEADS * 4)?),
            5 => bad.scale = f32::NAN,
            6 => bad.status = Some(fx.bufs[OUT].range(0, 4)?),
            7 => bad.status = Some(fx.bufs[PART].range(0, 4)?),
            8 => bad.status = Some(fx.bufs[Q].range(0, 4)?),
            9 => bad.status = Some(bad.k_cache.buffer().range(4, 4)?),
            _ => bad.status = Some(bad.v_cache.buffer().range(4, 4)?),
        }
        let result = fx.ctx.attn_decode_sinks_via_prefill(&fx.stream, bad);
        refusal_results.push(format!("{invalid}: {result:?}"));
        refusals &= matches!(result, Err(EngError::BadRequest(_)));
    }
    refusals &= before == [fx.bytes(PART), fx.bytes(OUT), fx.bytes(STATUS)];
    fx.reset_outputs()?;
    fx.ctx.attn_decode_sinks_via_prefill(&fx.stream, call()?)?;
    fx.stream.synchronize()?;
    let later = fx.out_values();
    let status = fx.status_word();
    let guards = inputs == [fx.bytes(Q), fx.bytes(KC), fx.bytes(VC), fx.bytes(PROJ)]
        && fx.bytes(OUT)[HEADS * DIM * 4..]
            .iter()
            .all(|&b| b == OUT_FILL)
        && fx.bytes(PART)[fx.base[PART].2..]
            .iter()
            .all(|&b| b == PART_FILL)
        && fx.bytes(STATUS)[4..].iter().all(|&b| b == 0)
        && fx
            .sink_buf
            .as_ref()
            .is_some_and(|b| testkit::read_bytes(&fx.ctx, b, b.bytes()) == le(&sinks))
        && buffers
            .as_ref()
            .is_none_or(|[kb, vb]| k.unchanged(&fx.ctx, kb) && v.unchanged(&fx.ctx, vb));
    drop((buffers, fx));
    Ok(Observation {
        actual,
        later,
        without_sink,
        reference,
        status,
        guards,
        refusals,
        refusal_results,
        distinct_epochs,
    })
}

#[test]
#[ignore = "requires an admitted CUDA correctness run; never silently skips a missing device"]
fn flint_sink_decode_m1_matches_fp64_and_repeats_bitwise() {
    let plan = KvPlan::new(1031, 1, true).unwrap();
    let deep = KvPlan::new(8199, 1, true).unwrap();
    let mut cases = Vec::new();
    for (layer, plan, p0) in (0..49)
        .map(|layer| (layer, &plan, 1026))
        .chain([0, 2].map(|layer| (layer, &deep, 8191)))
    {
        let rows = LayerKvAllocation::new(plan, layer).unwrap().cache_rows();
        let span = if rows == WINDOW {
            CpfAttnSpan {
                window: WINDOW as i64,
                ring: rows as i64,
            }
        } else {
            CpfAttnSpan::FULL
        };
        cases.push((layer, rows, span, p0));
    }
    cases.push((
        2,
        519,
        CpfAttnSpan {
            window: WINDOW as i64,
            ring: 519,
        },
        1026,
    ));
    for (layer, rows, span, p0) in cases {
        for storage in [
            AttnKvStorage::F16,
            AttnKvStorage::Q8Affine {
                group_size: 32,
                signed: true,
            },
        ] {
            let shape = AttnPrefillShape {
                m: 1,
                p0,
                n_head: HEADS,
                n_kv_head: KV_HEADS,
                head_dim: DIM,
                ld: 0,
                cache_rows: rows,
            };
            let got = observe(layer, shape, span, storage).expect("sink-aware m1 device phase");
            assert!(
                got.guards && got.refusals && got.distinct_epochs,
                "layer{layer} P{p0} rows{rows} storage{storage:?}: guards={} refusals={} epochs={} results={:?}",
                got.guards,
                got.refusals,
                got.distinct_epochs,
                got.refusal_results
            );
            assert_eq!(got.status, 0);
            let ratio = super::allocation::check_close(&got.actual, &got.reference);
            assert_eq!(
                got.actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                got.later.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_ne!(
                got.actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                got.without_sink
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "Flint sink decode correctness layer{layer} P{p0} rows{rows} storage{storage:?} error/bound={ratio} bitwise=true refusals=true"
            );
        }
    }
}
