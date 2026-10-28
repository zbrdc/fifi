use super::super::{AttnKvExtents, AttnKvLayout, AttnKvView, AttnScalarLayout, AttnStrides};
use super::*;
use crate::kernels::attention_reference::oracle::sdpa_f64;
use crate::tensor::Dtype;

mod flint_rows;
#[cfg(all(test, feature = "cuda-q8-reference"))]
mod reference_hoist;

const ORIGINS: [usize; 3] = [3, 8, 12];

struct Plane {
    layout: AttnKvLayout,
    extents: AttnKvExtents,
    bytes: [Vec<u8>; 3],
    values: Vec<f64>,
}

fn put_scalar(bytes: &mut [u8], offset: usize, dtype: Dtype, value: f32) {
    let encoded = match dtype {
        Dtype::F16 => f16_bits(value).to_le_bytes().to_vec(),
        Dtype::Bf16 => ((value.to_bits() >> 16) as u16).to_le_bytes().to_vec(),
        Dtype::F32 => value.to_le_bytes().to_vec(),
        _ => panic!("test scalar dtype"),
    };
    bytes[offset..offset + encoded.len()].copy_from_slice(&encoded);
}

fn scalar_layout(
    shape: AttnPrefillShape,
    dtype: Dtype,
    role: usize,
    group_size: usize,
) -> AttnScalarLayout {
    let width = dtype.byte_width() as usize;
    let groups = shape.head_dim / group_size;
    let group_stride_bytes = width * (role + 1);
    let head_stride_bytes = groups * group_stride_bytes + width * 2;
    AttnScalarLayout {
        dtype,
        strides: AttnStrides {
            time_stride_bytes: shape.n_kv_head * head_stride_bytes + width * 3,
            head_stride_bytes,
        },
        group_stride_bytes,
    }
}

// Exact binary fractions keep the independent decoded f64 values equal for all scalar dtypes.
fn plane(
    shape: AttnPrefillShape,
    storage: AttnKvStorage,
    scalar: (Dtype, Option<Dtype>),
    role: usize,
) -> Plane {
    let AttnKvStorage::Q8Affine { signed, group_size } = storage else {
        panic!("test storage")
    };
    let payload_head = shape.head_dim + role * 7 + 3;
    let layout = AttnKvLayout {
        storage,
        payload_strides: AttnStrides {
            time_stride_bytes: shape.n_kv_head * payload_head + 11,
            head_stride_bytes: payload_head,
        },
        scales: Some(scalar_layout(shape, scalar.0, role, group_size)),
        bias: scalar
            .1
            .map(|dtype| scalar_layout(shape, dtype, role + 1, group_size)),
    };
    let extents = layout
        .extents(shape.cache_rows, shape.n_kv_head, shape.head_dim)
        .unwrap();
    let mut bytes = [
        vec![0x63; ORIGINS[0] + extents.payload_bytes + SLACK],
        vec![0x53; ORIGINS[1] + extents.scale_bytes.unwrap() + SLACK],
        vec![0x43; ORIGINS[2] + extents.bias_bytes.unwrap_or(0) + SLACK],
    ];
    let mut values = vec![0.0; shape.cache_rows * shape.n_kv_head * shape.head_dim];
    for row in 0..shape.cache_rows {
        for h in 0..shape.n_kv_head {
            for g in 0..shape.head_dim / group_size {
                let scale = (1 + (row + h + g + role) % 4) as f32 / 128.0;
                let bias = ((row + 2 * h + g + role) % 5) as f32 / 32.0 - 1.0 / 16.0;
                for (index, parameter, value) in [(1, layout.scales, scale), (2, layout.bias, bias)]
                {
                    if let Some(parameter) = parameter {
                        let offset = ORIGINS[index]
                            + row * parameter.strides.time_stride_bytes
                            + h * parameter.strides.head_stride_bytes
                            + g * parameter.group_stride_bytes;
                        put_scalar(&mut bytes[index], offset, parameter.dtype, value);
                    }
                }
                for lane in 0..group_size {
                    let d = g * group_size + lane;
                    let code = ((row * 17 + h * 23 + d * 13 + role * 31) % 256) as u8;
                    let offset = ORIGINS[0]
                        + row * layout.payload_strides.time_stride_bytes
                        + h * layout.payload_strides.head_stride_bytes
                        + d;
                    bytes[0][offset] = code;
                    let code = if signed {
                        f64::from(code as i8)
                    } else {
                        f64::from(code)
                    };
                    values[(row * shape.n_kv_head + h) * shape.head_dim + d] = f64::from(scale)
                        * code
                        + if scalar.1.is_some() {
                            f64::from(bias)
                        } else {
                            0.0
                        };
                }
            }
        }
    }
    Plane {
        layout,
        extents,
        bytes,
        values,
    }
}

impl Plane {
    fn upload(&self, ctx: &CudaContext) -> Result<[DeviceBuffer; 3], EngError> {
        Ok([
            ctx.upload(&self.bytes[0])?,
            ctx.upload(&self.bytes[1])?,
            ctx.upload(&self.bytes[2])?,
        ])
    }

    fn view<'a>(&self, buffers: &'a [DeviceBuffer; 3]) -> Result<AttnKvView<'a>, EngError> {
        Ok(AttnKvView {
            layout: self.layout,
            payload: buffers[0].range(ORIGINS[0], self.extents.payload_bytes)?,
            scales: self
                .extents
                .scale_bytes
                .map(|n| buffers[1].range(ORIGINS[1], n))
                .transpose()?,
            bias: self
                .extents
                .bias_bytes
                .map(|n| buffers[2].range(ORIGINS[2], n))
                .transpose()?,
        })
    }

    fn unchanged(&self, ctx: &CudaContext, buffers: &[DeviceBuffer; 3]) -> bool {
        self.bytes
            .iter()
            .zip(buffers)
            .all(|(expected, buffer)| testkit::read_bytes(ctx, buffer, buffer.bytes()) == *expected)
    }
}

#[derive(Clone, Copy)]
struct Case {
    shape: AttnPrefillShape,
    signed: bool,
    group_size: usize,
    scalar: (Dtype, Option<Dtype>),
    span: CpfAttnSpan,
    splits: usize,
    sinks: bool,
    gate: bool,
    segmented: bool,
}

struct Observation {
    actual: Vec<f32>,
    reference: Vec<f64>,
    bound: f64,
    status: i32,
    guards: bool,
    refusals: bool,
}

// Assertions run only after this Result phase has dropped every device owner.
fn observe(case: Case) -> Result<Observation, EngError> {
    let shape = case.shape;
    let storage = AttnKvStorage::Q8Affine {
        group_size: case.group_size,
        signed: case.signed,
    };
    let k = plane(shape, storage, case.scalar, 0);
    let v = plane(shape, storage, case.scalar, 1);
    let q_len = shape.m * shape.n_head * shape.head_dim;
    let kv_len = shape.cache_rows * shape.n_kv_head * shape.head_dim;
    let mut input = Data {
        q: (0..q_len)
            .map(|i| ((i % 31) as f32 - 15.0) / 256.0)
            .collect(),
        k: vec![0; kv_len],
        v: vec![0; kv_len],
        proj: vec![0.25; shape.m * shape.ld],
    };
    if case.group_size == 256 {
        input.q[0] = 70_000.0; // Finite F32, not F16: catch a hidden query downcast.
    }
    let split = CpfAttnSplit {
        max_splits: case.splits,
        min_keys: 32,
    };
    let mut fx = Fx::with_data(shape, split, case.gate, true, &input)?;
    fx.span = case.span;
    let [kb, vb] = [k.upload(&fx.ctx)?, v.upload(&fx.ctx)?];
    let views = [k.view(&kb)?, v.view(&vb)?];
    let sinks: Vec<f32> = (0..shape.n_head).map(|h| h as f32 / 16.0 - 1.0).collect();
    if case.sinks {
        fx.sink_buf = Some(fx.ctx.upload(&le(&sinks))?);
    }
    let call = || -> Result<AttnPrefillCall<'_>, EngError> {
        let mut call = fx.call()?;
        call.storage = storage;
        call.k_cache = views[0].payload;
        call.v_cache = views[1].payload;
        call.kv_views = Some(views);
        Ok(call)
    };
    let before = [fx.bytes(PART), fx.bytes(OUT), fx.bytes(STATUS)];
    let mut refusals = true;
    for invalid in 0..6 {
        let mut bad = call()?;
        match invalid {
            0 => bad.kv_views = None,
            1 => {
                let mut broken = views;
                broken[0].scales = None;
                bad.kv_views = Some(broken);
            }
            2 => {
                let mut broken = views;
                broken[0].scales = Some(kb[1].range(
                    ORIGINS[1],
                    k.extents.scale_bytes.unwrap_or(0).saturating_sub(1),
                )?);
                bad.kv_views = Some(broken);
            }
            3 => {
                let mut broken = views;
                broken[0].scales =
                    Some(fx.bufs[PART].range(0, k.extents.scale_bytes.unwrap_or(0))?);
                bad.kv_views = Some(broken);
            }
            4 => bad.status = Some(fx.bufs[OUT].range(0, 4)?),
            _ => bad.storage = AttnKvStorage::F16,
        }
        refusals &= matches!(
            fx.ctx.attn_prefill(&fx.stream, bad),
            Err(EngError::BadRequest(_))
        );
    }
    refusals &= before == [fx.bytes(PART), fx.bytes(OUT), fx.bytes(STATUS)];
    if case.segmented {
        fx.ctx.attn_prefill_part(
            &fx.stream,
            call()?,
            AttnPrefillSegment {
                key_origin: 0,
                key_rows: shape.cache_rows,
                part_index: 0,
            },
        )?;
        fx.ctx.attn_prefill_combine(&fx.stream, call()?)?;
    } else {
        fx.ctx.attn_prefill(&fx.stream, call()?)?;
    }
    fx.stream.synchronize()?;
    let actual = fx.out_values();
    let status = fx.status_word();
    let guards = fx.bytes(OUT)[q_len * 4..].iter().all(|&b| b == OUT_FILL)
        && fx.bytes(PART)[fx.base[PART].2..]
            .iter()
            .all(|&b| b == PART_FILL)
        && k.unchanged(&fx.ctx, &kb)
        && v.unchanged(&fx.ctx, &vb)
        && fx.bytes(Q)[..q_len * 4] == le(&input.q)
        && fx.bytes(PROJ)[..input.proj.len() * 4] == le(&input.proj)
        && fx.bytes(STATUS)[4..].iter().all(|&b| b == 0)
        && fx
            .sink_buf
            .as_ref()
            .is_none_or(|b| testkit::read_bytes(&fx.ctx, b, b.bytes()) == le(&sinks));
    drop((kb, vb, fx));
    let q: Vec<f64> = input.q.iter().copied().map(f64::from).collect();
    let sink_ref: Vec<f64> = sinks.iter().copied().map(f64::from).collect();
    let mut reference = sdpa_f64(
        &q,
        &k.values,
        &v.values,
        shape.m,
        shape.n_head,
        shape.cache_rows,
        shape.head_dim,
        shape.n_head / shape.n_kv_head,
        f64::from(SCALE),
        shape.p0 as u32,
        0,
        (case.span.window > 0).then_some(case.span.window as u32),
        case.sinks.then_some(sink_ref.as_slice()),
    );
    if case.gate {
        let sigmoid = 1.0 / (1.0 + (-0.25_f64).exp());
        reference.iter_mut().for_each(|value| *value *= sigmoid);
    }
    // FP32 Q/dequant/accumulation: fixed 8x tighter than the existing F16 PA1 bound.
    let bound = 2f64.powi(-12) * v.values.iter().map(|v| v.abs()).fold(1.0, f64::max);
    Ok(Observation {
        actual,
        reference,
        bound,
        status,
        guards,
        refusals,
    })
}

#[test]
fn q8_affine_rows_match_f64_and_refuse_invalid_views_without_poisoning() {
    if !testkit::device_present(
        "q8_affine_rows_match_f64_and_refuse_invalid_views_without_poisoning",
    ) {
        return;
    }
    let mut cases = Vec::new();
    for (dim, group, group_size, signed, scalar, bias) in [
        (64, 3, 16, true, Dtype::F16, Some(Dtype::F32)),
        (128, 4, 64, false, Dtype::Bf16, Some(Dtype::F16)),
        (256, 1, 128, true, Dtype::F32, Some(Dtype::Bf16)),
        // Q8_0-equivalent separated planes; NOT the raw interleaved 34-byte GGUF layout.
        (128, 8, 32, true, Dtype::F16, None),
        (64, 1, 1, true, Dtype::F32, None),
        (256, 3, 256, false, Dtype::Bf16, Some(Dtype::F16)),
    ] {
        for (window, segmented) in [(0, false), (19, false), (0, true)] {
            let shape = AttnPrefillShape {
                m: 17,
                n_head: 2 * group,
                n_kv_head: 2,
                head_dim: dim,
                cache_rows: 65,
                ld: 2 * group * 2 * dim,
                p0: 48,
            };
            cases.push(Case {
                shape,
                signed,
                group_size,
                scalar: (scalar, bias),
                span: CpfAttnSpan { window, ring: 0 },
                splits: if segmented { 1 } else { 4 },
                sinks: group != 8,
                gate: group != 8,
                segmented,
            });
        }
    }
    // More than 64 active splits: D64 must load/reduce all 128 statistic lanes.
    cases.push(Case {
        shape: AttnPrefillShape {
            m: 1,
            n_head: 3,
            n_kv_head: 1,
            head_dim: 64,
            cache_rows: 4097,
            ld: 0,
            p0: 4096,
        },
        signed: true,
        group_size: 32,
        scalar: (Dtype::F16, None),
        span: CpfAttnSpan::FULL,
        splits: 128,
        sinks: false,
        gate: false,
        segmented: false,
    });
    for case in cases {
        let result = observe(case).expect("Q8 affine device phase");
        assert_eq!(result.status, 0);
        assert!(
            result.guards && result.refusals,
            "D{}/GQA{}",
            case.shape.head_dim,
            case.shape.n_head / case.shape.n_kv_head
        );
        assert_eq!(result.actual.len(), result.reference.len());
        let worst = result
            .actual
            .iter()
            .zip(&result.reference)
            .map(|(&actual, &reference)| {
                if actual.is_finite() {
                    (f64::from(actual) - reference).abs()
                } else {
                    f64::INFINITY
                }
            })
            .fold(0.0, f64::max);
        assert!(
            worst <= result.bound,
            "D{}/GQA{}/S{}: {worst} > {}",
            case.shape.head_dim,
            case.shape.n_head / case.shape.n_kv_head,
            case.splits,
            result.bound
        );
        eprintln!(
            "Q8_CASE D{} H{} T{} splits{} window{} segmented{} max_delta={worst}",
            case.shape.head_dim,
            case.shape.n_head,
            case.shape.m,
            case.splits,
            case.span.window,
            case.segmented
        );
    }
}
