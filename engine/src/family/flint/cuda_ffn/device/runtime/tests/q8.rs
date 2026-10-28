use super::*;
use crate::kernels::attention_reference::oracle::{sdpa_f64, ulp_bf16};

struct WriterPlane {
    input: Vec<f32>,
    codes: Vec<u8>,
    scales: Vec<u8>,
    decoded: Vec<f64>,
}

fn writer_plane(factor: f32) -> WriterPlane {
    let input: Vec<_> = (0_usize..8 * 512)
        .map(|i| {
            let (row, head, group, lane) = (i / 512, i % 512 / 128, i % 128 / 32, i % 32);
            let amplitude = 0.25 * (row * 16 + head * 4 + group + 1) as f32;
            let magnitude = ((lane + head * 5 + row * 3 + group * 7) % 32 + 1) as f32 / 32.0;
            let sign = if (lane + head + row).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            half::bf16::from_f32(factor * amplitude * magnitude * sign).to_f32()
        })
        .collect();
    let mut codes = Vec::with_capacity(input.len());
    let mut scales = Vec::with_capacity(input.len() / 16);
    let mut decoded = vec![0.0; 527 * 512];
    for values in input.as_chunks::<32>().0 {
        let amax = values.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let scale = half::f16::from_f32(amax / 127.0);
        scales.extend_from_slice(&scale.to_bits().to_ne_bytes());
        for &value in values {
            // StoredHalf uses rounded division and nearest-even integer conversion.
            let code = (value / scale.to_f32())
                .round_ties_even()
                .clamp(-127.0, 127.0) as i8;
            decoded[519 * 512 + codes.len()] = f64::from(f32::from(code) * scale.to_f32());
            codes.push(code as u8);
        }
    }
    assert_ne!(
        codes[..128],
        codes[128..256],
        "head permutation must be observable"
    );
    assert_ne!(
        codes[..512],
        codes[512..1024],
        "time permutation must be observable"
    );
    WriterPlane {
        input,
        codes,
        scales,
        decoded,
    }
}

fn check_written_plane(
    stream: &DeviceStream,
    view: AttnKvView<'_>,
    source: &WriterPlane,
    cells: usize,
) {
    let extents = view.layout.extents(cells, 4, 128).unwrap();
    assert_eq!(extents.payload_bytes, cells * 512);
    assert_eq!(extents.scale_bytes, Some(cells * 32));
    assert_eq!(extents.bias_bytes, None);
    let scalar = view.layout.scales.unwrap();
    let mut expected_codes = vec![0x55; cells * 512];
    let mut expected_scales = vec![0xff; cells * 32];
    for row in 0..8 {
        let cell = (519 + row) % cells;
        for head in 0..4 {
            let from = (row * 4 + head) * 128;
            let to = head * view.layout.payload_strides.head_stride_bytes
                + cell * view.layout.payload_strides.time_stride_bytes;
            expected_codes[to..to + 128].copy_from_slice(&source.codes[from..from + 128]);
            for group in 0..4 {
                let from = ((row * 4 + head) * 4 + group) * 2;
                let to = head * scalar.strides.head_stride_bytes
                    + cell * scalar.strides.time_stride_bytes
                    + group * scalar.group_stride_bytes;
                expected_scales[to..to + 2].copy_from_slice(&source.scales[from..from + 2]);
            }
        }
    }
    for (range, expected) in [
        (view.payload, expected_codes),
        (view.scales.unwrap(), expected_scales),
    ] {
        let mut actual = vec![0; range.bytes()];
        range
            .buffer()
            .read_at(stream, range.offset(), &mut actual)
            .unwrap();
        assert!(
            actual == expected,
            "Q8 plane bytes or unused poison cells changed"
        );
    }
}

fn attention_error(actual: &[f32], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut worst = 0.0_f64;
    for (actual, expected) in actual
        .as_chunks::<128>()
        .0
        .iter()
        .zip(expected.as_chunks::<128>().0)
    {
        let magnitude = expected.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let bound = 2.0 * ulp_bf16(magnitude);
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(actual.is_finite() && expected.is_finite());
            worst = worst.max((f64::from(actual) - expected).abs() / bound);
        }
    }
    worst
}

fn check_writer_sink(
    group: &Group<'_, '_>,
    views: [AttnKvView<'_>; 2],
    source: &[WriterPlane; 2],
    sliding: bool,
) {
    let (ctx, stream) = (group.ctx, group.stream);
    let cells = views[0].payload.bytes() / 512;
    let first = if sliding { 7 } else { 0 };
    // Seed valid old history only after the whole-plane writer/poison checks. The ABI,
    // not the view under test, defines these head-major addresses and the reference.
    for view in views {
        for head in 0..4 {
            let offset = (head * cells + first) * 128;
            view.payload
                .buffer()
                .write_at(
                    stream,
                    view.payload.offset() + offset,
                    &vec![0; (519 - first) * 128],
                )
                .unwrap();
            let scales = view.scales.unwrap();
            scales
                .buffer()
                .write_at(
                    stream,
                    scales.offset() + (head * cells + first) * 8,
                    &half::f16::ONE
                        .to_bits()
                        .to_ne_bytes()
                        .repeat((519 - first) * 4),
                )
                .unwrap();
        }
    }
    let sinks: Vec<f32> = (0..32)
        .map(|head| [-4.0, 0.5, 12.0][head % 3] + head as f32 / 64.0)
        .collect();
    let sink_buffer = ctx
        .upload(
            &sinks
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let sinks_f64: Vec<_> = sinks.iter().copied().map(f64::from).collect();
    let span = CpfAttnSpan {
        window: if sliding { 513 } else { 0 },
        ring: if sliding { cells as i64 } else { 0 },
    };
    let mut swapped = views;
    for (correct, wrong) in views.iter().zip(&mut swapped) {
        wrong.layout.payload_strides = AttnStrides {
            time_stride_bytes: 512,
            head_stride_bytes: 128,
        };
        assert_eq!(
            correct.layout.extents(cells, 4, 128).unwrap().payload_bytes,
            wrong.layout.extents(cells, 4, 128).unwrap().payload_bytes,
            "the negative control must pass the same extent check",
        );
    }
    for row in 0..8 {
        let position = 519 + row;
        let q: Vec<f32> = (0..4096)
            .map(|i| ((i * 7 + row * 5) % 31) as f32 / 512.0 - 15.0 / 512.0)
            .collect();
        let q_f64: Vec<_> = q.iter().copied().map(f64::from).collect();
        let scale = 128.0_f32.sqrt().recip();
        let expected = sdpa_f64(
            &q_f64,
            &source[0].decoded,
            &source[1].decoded,
            1,
            32,
            527,
            128,
            8,
            f64::from(scale),
            position as u32,
            0,
            sliding.then_some(513),
            Some(&sinks_f64),
        );
        let query = ctx
            .upload(&q.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>())
            .unwrap();
        let plan = cpf_attn_launch_plan(
            HotScratch::geometry(),
            position + 1,
            span.window as usize,
            CPF_ATTN_SM_COUNT,
        )
        .unwrap();
        let part_bytes = cpf_attn_part_bytes(HotScratch::geometry(), plan).unwrap();
        let part = ctx.upload(&vec![0xff; part_bytes]).unwrap();
        let output = ctx.upload(&[0xff; 4096 * 4]).unwrap();
        let status = ctx.upload(&0_u32.to_ne_bytes()).unwrap();
        for (mutated, kv_views) in [(false, views), (true, swapped)] {
            output.write_at(stream, 0, &[0xff; 4096 * 4]).unwrap();
            part.write_at(stream, 0, &vec![0xff; part_bytes]).unwrap();
            status.write_at(stream, 0, &0_u32.to_ne_bytes()).unwrap();
            ctx.attn_decode_sinks_via_prefill(
                stream,
                AttnPrefillCall {
                    storage: kv_views[0].layout.storage,
                    kv_views: Some(kv_views),
                    shape: AttnPrefillShape {
                        m: 1,
                        n_head: 32,
                        n_kv_head: 4,
                        head_dim: 128,
                        cache_rows: cells,
                        ld: 0,
                        p0: position as i64,
                    },
                    span,
                    split: CpfAttnSplit {
                        max_splits: plan.max_splits,
                        min_keys: plan.min_keys,
                    },
                    q_tile: plan.q_tile,
                    scale,
                    q: query.range(0, query.bytes()).unwrap(),
                    proj: None,
                    k_cache: kv_views[0].payload,
                    v_cache: kv_views[1].payload,
                    part: part.range(0, part_bytes).unwrap(),
                    out: output.range(0, 4096 * 4).unwrap(),
                    status: Some(status.range(0, 4).unwrap()),
                    sinks: Some(sink_buffer.range(0, 32 * 4).unwrap()),
                },
            )
            .unwrap();
            stream.synchronize().unwrap();
            let mut flag = [0; 4];
            status.read_at(stream, 0, &mut flag).unwrap();
            assert_eq!(u32::from_ne_bytes(flag), 0);
            let actual = values(stream, output.range(0, 4096 * 4).unwrap()).unwrap();
            let error = attention_error(&actual, &expected);
            if mutated {
                assert!(
                    error > 1.0,
                    "stride mutation was vacuous: sliding={sliding} P{position} error/bound={error}"
                );
            } else {
                assert!(
                    error <= 1.0,
                    "writer-to-sink CPF mismatch: sliding={sliding} P{position} error/bound={error}"
                );
            }
            eprintln!(
                "Q8 writer_to_sink_CPF sliding={sliding} P{position} stride_mutation={mutated} error/bound={error}"
            );
        }
    }
}

fn writer_sink_case(ctx: &CudaContext, stream: &DeviceStream, layer: usize) {
    let host_writes = test_counts()[1];
    let q8_launches = TEST_Q8_CHUNK_LAUNCHES.with(std::cell::Cell::get);
    let plan = KvPlan::new(2048, 8, true)
        .unwrap()
        .with_storage(KvStorage::Q8G32);
    let mut cache = LayerKvAllocation::new(&plan, layer)
        .unwrap()
        .allocate(ctx)
        .unwrap();
    let cells = plan.cache_rows(layer).unwrap();
    let source = [writer_plane(1.0), writer_plane(-2.0)];
    let inputs = source.each_ref().map(|plane| {
        ctx.upload(
            &plane
                .input
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    });
    for plane in [&cache.keys, &cache.values] {
        plane
            .write_at(stream, 0, &vec![0x55; plane.bytes()])
            .unwrap();
    }
    for plane in cache.scales.as_ref().unwrap() {
        plane
            .write_at(stream, 0, &vec![0xff; plane.bytes()])
            .unwrap();
    }
    while cache.cursor.committed_prefix() < 519 {
        let position = cache.cursor.committed_prefix();
        cache
            .cursor
            .begin_append(position, 8.min(519 - position))
            .unwrap()
            .commit();
    }
    let append = cache.cursor.begin_append(519, 8).unwrap();
    let group = Group::new(ctx, stream, None, 0).unwrap();
    let planes = [&cache.keys, &cache.values];
    assert!(matches!(
        write_cache(&group, [&inputs[0], &inputs[1]], (planes, None, &append), 8),
        Err(EngError::BadRequest(message)) if message == "Flint CUDA FFN: Flint cache storage and scale planes disagree"
    ));
    let mut f16_cursor = LayerKvAllocation::new(&KvPlan::new(2048, 8, true).unwrap(), layer)
        .unwrap()
        .cursor();
    let f16_append = f16_cursor.begin_append(0, 8).unwrap();
    assert!(matches!(
        write_cache(&group, [&inputs[0], &inputs[1]], (planes, cache.scales.as_ref(), &f16_append), 8),
        Err(EngError::BadRequest(message)) if message == "Flint CUDA FFN: Flint cache storage and scale planes disagree"
    ));
    assert!(matches!(
        write_cache(&group, [&inputs[0], &inputs[1]], (planes, cache.scales.as_ref(), &append), 7),
        Err(EngError::BadRequest(message)) if message == "Flint CUDA FFN: Flint cache append runs do not cover source rows"
    ));
    let views = write_cache(
        &group,
        [&inputs[0], &inputs[1]],
        (planes, cache.scales.as_ref(), &append),
        8,
    )
    .unwrap()
    .unwrap();
    for (view, source) in views.into_iter().zip(&source) {
        check_written_plane(stream, view, source, cells);
    }
    check_writer_sink(&group, views, &source, layer == 2);
    stream.synchronize().unwrap();
    append.commit();
    assert_eq!(cache.cursor.committed_prefix(), 527);
    assert_eq!(test_counts()[1], host_writes);
    assert_eq!(
        TEST_Q8_CHUNK_LAUNCHES.with(std::cell::Cell::get),
        q8_launches + if layer == 2 { 2 } else { 1 }
    );
}

#[test]
fn device_q8_wrap_has_zero_host_kv_writes_and_two_device_launches() {
    if !crate::kernels::cuda::device_present("Flint Q8 writer-to-sink wrap fixture") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    writer_sink_case(&ctx, &ctx.stream().unwrap(), 2);
}

#[test]
fn device_q8_full_layer_writer_views_match_sink_reference() {
    if !crate::kernels::cuda::device_present("Flint Q8 writer-to-sink full-layer fixture") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    writer_sink_case(&ctx, &ctx.stream().unwrap(), 0);
}

const REFUSED_BOUNDARY_VALUES: [f32; 8] = [
    f32::from_bits(1.0_f32.to_bits() + 1),
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    65536.0,
    -65536.0,
    8_388_608.0,
    -8_388_608.0,
];

fn boundary_cache(group: &Group<'_, '_>) -> LayerKv {
    let plan = KvPlan::new(2048, 8, true)
        .unwrap()
        .with_storage(KvStorage::Q8G32);
    let mut cache = LayerKvAllocation::new(&plan, 2)
        .unwrap()
        .allocate(group.ctx)
        .unwrap();
    for (plane, poison) in [
        (&cache.keys, 0x55),
        (&cache.values, 0x55),
        (&cache.scales.as_ref().unwrap()[0], 0xff),
        (&cache.scales.as_ref().unwrap()[1], 0xff),
    ] {
        plane
            .write_at(group.stream, 0, &vec![poison; plane.bytes()])
            .unwrap();
    }
    while cache.cursor.committed_prefix() < 519 {
        let position = cache.cursor.committed_prefix();
        cache
            .cursor
            .begin_append(position, 8.min(519 - position))
            .unwrap()
            .commit();
    }
    cache
}

fn check_refused_boundary(group: &Group<'_, '_>, invalid_plane: usize, value: f32) {
    let mut cache = boundary_cache(group);
    let mut input = [vec![1.0_f32; 8 * 512], vec![-1.0_f32; 8 * 512]];
    input[invalid_plane][8 * 512 - 1] = value;
    let inputs = input.each_ref().map(|values| {
        group
            .ctx
            .upload(
                &values
                    .iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
    });
    let counts = test_counts();
    let launches = TEST_Q8_CHUNK_LAUNCHES.with(std::cell::Cell::get);
    {
        let append = cache.cursor.begin_append(519, 8).unwrap();
        let error = write_cache(
            group,
            [&inputs[0], &inputs[1]],
            ([&cache.keys, &cache.values], cache.scales.as_ref(), &append),
            8,
        )
        .err()
        .expect("invalid Q8 writer input must be refused");
        assert_eq!(
            error,
            EngError::ModelOutput(
                "Flint invalid BF16 input or overflow in physical KV cache".into()
            ),
            "plane={invalid_plane} value={value:?}"
        );
    }
    group.stream.synchronize().unwrap();
    assert_eq!(test_counts(), counts);
    assert_eq!(TEST_Q8_CHUNK_LAUNCHES.with(std::cell::Cell::get), launches);
    assert_eq!(cache.cursor.committed_prefix(), 519);
    assert!(cache.cursor.test_poisoned());
    for (plane, poison) in [
        (&cache.keys, 0x55),
        (&cache.values, 0x55),
        (&cache.scales.as_ref().unwrap()[0], 0xff),
        (&cache.scales.as_ref().unwrap()[1], 0xff),
    ] {
        let mut actual = vec![0; plane.bytes()];
        plane.read_at(group.stream, 0, &mut actual).unwrap();
        assert!(
            actual.iter().all(|&byte| byte == poison),
            "refusal changed a Q8 plane: input={invalid_plane} value={value:?}"
        );
    }
    assert!(cache.cursor.begin_append(519, 8).is_err());
    eprintln!(
        "Q8 numerical refusal plane={invalid_plane} value={value:?} launches=0 planes_unchanged=4 prefix=519 poisoned=true"
    );
}

#[test]
fn device_q8_writer_refuses_non_bf16_nonfinite_and_overflow_without_writes() {
    if !crate::kernels::cuda::device_present("Flint Q8 writer numerical refusal fixture") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let group = Group::new(&ctx, &stream, None, 0).unwrap();
    for invalid_plane in 0..2 {
        for value in REFUSED_BOUNDARY_VALUES {
            check_refused_boundary(&group, invalid_plane, value);
        }
    }
}

fn stored_half_boundary_plane(factor: f32) -> WriterPlane {
    let unit = 2.0_f32.powi(-25);
    // At half the least F16 scale, ties-to-even stores zero; the next BF16 amax stores 2^-24.
    let pattern = [-1.0, 1.0, 0.5, 0.0, -1.0, 1.0, 0.5, 0.0];
    let groups = [
        (pattern.map(|x| x * 0.0), 0_u16, [0_i8; 8]),
        (pattern.map(|x| x * (126.0 * unit)), 0, [0; 8]),
        (pattern.map(|x| x * (127.0 * unit)), 0, [0; 8]),
        (
            pattern.map(|x| x * (128.0 * unit)),
            1,
            [-64, 64, 32, 0, -64, 64, 32, 0],
        ),
        (
            pattern.map(|x| x * 65280.0),
            0x6004,
            [-127, 127, 64, 0, -127, 127, 64, 0],
        ),
        (
            [-127.0, 127.0, -0.5, 0.5, -2.5, 2.5, 0.0, 0.0],
            0x3c00,
            [-127, 127, 0, 0, -2, 2, 0, 0],
        ),
        (
            pattern.map(|x| x * (190.0 * 2.0_f32.powi(-24))),
            1,
            [-127, 127, 95, 0, -127, 127, 95, 0],
        ),
    ];
    let input: Vec<_> = (0..8 * 512)
        .map(|i| factor * groups[(i / 32) % groups.len()].0[i % 8])
        .collect();
    assert!(input.iter().all(|x| x.to_bits() & 0xffff == 0));
    let codes = (0..input.len())
        .map(|i| (factor as i8 * groups[(i / 32) % groups.len()].2[i % 8]) as u8)
        .collect();
    let scales = (0..input.len() / 32)
        .flat_map(|i| groups[i % groups.len()].1.to_ne_bytes())
        .collect();
    WriterPlane {
        input,
        codes,
        scales,
        decoded: Vec::new(),
    }
}

#[test]
fn device_q8_writer_pins_zero_tiny_and_maximum_stored_half_groups() {
    if !crate::kernels::cuda::device_present("Flint Q8 writer StoredHalf boundary fixture") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let group = Group::new(&ctx, &stream, None, 0).unwrap();
    let mut cache = boundary_cache(&group);
    let source = [
        stored_half_boundary_plane(1.0),
        stored_half_boundary_plane(-1.0),
    ];
    let inputs = source.each_ref().map(|plane| {
        ctx.upload(
            &plane
                .input
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    });
    let host_writes = test_counts()[1];
    let launches = TEST_Q8_CHUNK_LAUNCHES.with(std::cell::Cell::get);
    let append = cache.cursor.begin_append(519, 8).unwrap();
    let views = write_cache(
        &group,
        [&inputs[0], &inputs[1]],
        ([&cache.keys, &cache.values], cache.scales.as_ref(), &append),
        8,
    )
    .unwrap()
    .unwrap();
    for (view, expected) in views.into_iter().zip(&source) {
        check_written_plane(&stream, view, expected, 520);
    }
    stream.synchronize().unwrap();
    append.commit();
    assert_eq!(cache.cursor.committed_prefix(), 527);
    assert!(!cache.cursor.test_poisoned());
    assert_eq!(test_counts()[1], host_writes);
    assert_eq!(
        TEST_Q8_CHUNK_LAUNCHES.with(std::cell::Cell::get),
        launches + 2
    );
    eprintln!(
        "Q8 StoredHalf zero/below-tie/tie/above-tie/maximum/code-RNE-ties/subnormal-saturation groups byte-exact K+V launches=2 prefix=527"
    );
}

#[test]
fn q8_writer_boundary_inputs_pin_host_refusals_and_stored_half_goldens() {
    for value in REFUSED_BOUNDARY_VALUES {
        assert_eq!(
            f16_row(&[value; 512]).unwrap_err(),
            EngError::ModelOutput(
                "Flint invalid BF16 input or overflow in physical KV cache".into()
            )
        );
    }
    for factor in [1.0, -1.0] {
        let source = stored_half_boundary_plane(factor);
        for row in source.input.as_chunks::<512>().0 {
            assert_eq!(f16_row(row).unwrap().len(), 1024);
        }
        for (index, group) in source.input.as_chunks::<32>().0.iter().enumerate() {
            let amax = group.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            let scale = half::f16::from_f32(amax / 127.0);
            assert_eq!(
                source.scales[index * 2..index * 2 + 2],
                scale.to_bits().to_ne_bytes()
            );
            for (lane, value) in group.iter().enumerate() {
                let code = if scale.to_bits() == 0 {
                    0
                } else {
                    (value / scale.to_f32())
                        .round_ties_even()
                        .clamp(-127.0, 127.0) as i8
                };
                assert_eq!(
                    source.codes[index * 32 + lane],
                    code as u8,
                    "factor={factor} group={index} lane={lane}"
                );
            }
        }
    }
}
