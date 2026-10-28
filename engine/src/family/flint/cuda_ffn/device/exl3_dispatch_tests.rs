use super::*;
use crate::family::flint::exl3_resident::{
    FlintExpertView, FlintFfnView, FlintResidentLimits, load_flint_fixture,
};
use crate::family::project::cuda::{Reserve, exl3::Exl3Weight};
use crate::kernels::cuda::device_present;
use crate::weights::exl3::{Codebook, flint_pack::PackBuilder};
use half::{bf16, f16};

pub(super) fn upload_f32(ctx: &CudaContext, values: &[f32]) -> DeviceBuffer {
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    ctx.upload(&bytes).unwrap()
}

pub(super) fn weight(ctx: &CudaContext, shape: [usize; 2], bias: f32) -> Exl3Weight {
    let [k, n] = shape;
    Exl3Weight {
        trellis: ctx.upload(&vec![0; k * n / 256 * 3 * 16]).unwrap(),
        suh: upload_f32(ctx, &vec![1.0; k]),
        svh: upload_f32(ctx, &vec![1.0; n]),
        bias: Some(upload_f32(ctx, &vec![bias; n])),
        shape,
        bits_twice: 3,
        book: Codebook::Mul1,
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn assert_rows_match_serial<const N: usize>(
    ctx: &CudaContext,
    rows: &[[f32; N]],
    run: impl Fn(DeviceRange<'_>) -> DeviceBuffer,
) {
    assert!(rows.len() >= 2);
    let mut expected = Vec::new();
    for row in rows {
        let x = upload_f32(ctx, row);
        expected.extend(read(ctx, &run(x.range(0, x.bytes()).unwrap())));
    }
    assert!(expected.iter().all(|value| value.is_finite()));
    assert!(expected.len().is_multiple_of(rows.len()));
    let columns = expected.len() / rows.len();
    assert_ne!(
        &expected[..columns],
        &expected[columns..2 * columns],
        "fixture must expose row permutation"
    );
    let x = upload_f32(ctx, rows.as_flattened());
    let actual = run(x.range(0, x.bytes()).unwrap());
    assert_eq!(bits(&read(ctx, &actual)), bits(&expected));
}

fn read(ctx: &CudaContext, output: &DeviceBuffer) -> Vec<f32> {
    let mut bytes = vec![0; output.bytes()];
    ctx.download(output, &mut bytes).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_ne_bytes(*word))
        .collect()
}

#[test]
fn serving_projection_arithmetic_is_model_output_and_recovers() {
    if !device_present("serving_projection_arithmetic_is_model_output_and_recovers") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let weight = weight(&ctx, [256, 128], 0.25);
    let view = weight.view().unwrap();
    let project = |x: &DeviceBuffer| {
        super::exl3::project(&ctx, &stream, view, x.range(0, x.bytes()).unwrap())
            .map_err(super::runtime::model_forward_error)
    };
    let invalid = upload_f32(&ctx, &[f32::NAN; 256]);
    assert!(matches!(project(&invalid), Err(EngError::ModelOutput(_))));
    ctx.ensure_usable().unwrap();
    let valid = upload_f32(&ctx, &[0.0; 256]);
    assert!(
        read(&ctx, &project(&valid).unwrap())
            .iter()
            .all(|v| v.is_finite())
    );
    let wrong_shape = upload_f32(&ctx, &[0.0; 255]);
    assert!(matches!(
        super::exl3::project(
            &ctx,
            &stream,
            view,
            wrong_shape.range(0, wrong_shape.bytes()).unwrap()
        ),
        Err(EngError::BadRequest(_))
    ));
}

#[test]
fn serving_router_status_is_model_output_and_recovers() {
    if !device_present("serving_router_status_is_model_output_and_recovers") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let input = upload_f32(&ctx, &[0.0; 128]);
    let router = upload_f32(&ctx, &[0.0; TOP_K * 128]);
    let run = |status: u32| {
        let mut frame = Frame::new(&ctx, &stream)?;
        let x = frame.input(
            input.range(0, input.bytes())?,
            RowShape {
                rows: 1,
                columns: 128,
            },
        )?;
        let flag = frame.flag()?;
        flag.buffer()
            .write_at(&stream, flag.offset(), &status.to_le_bytes())?;
        let block = FfnBlock::new(
            1,
            FfnShape::Moe {
                hidden: 128,
                intermediate: 128,
                shared_intermediate: 128,
                experts: TOP_K,
            },
        )?;
        route(&mut frame, router.range(0, router.bytes())?, x, block)
            .map_err(super::runtime::model_forward_error)?;
        Ok::<(), EngError>(())
    };
    assert!(matches!(run(8), Err(EngError::ModelOutput(_))));
    ctx.ensure_usable().unwrap();
    assert!(run(0).is_ok());
}

#[test]
fn exl3_projection_returns_raw_f32_and_rejects_unhealthy_output() {
    if !device_present("exl3_projection_returns_raw_f32_and_rejects_unhealthy_output") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let bias = f32::from_bits(0x3e00_1234);
    assert_ne!(bias, bf16::from_f32(bias).to_f32());
    let weight = weight(&ctx, [256, 128], bias);
    let view = weight.view().unwrap();
    for value in [0.0, f32::NAN, 0.0] {
        let x = upload_f32(&ctx, &[value; 256]);
        let before = ctx.scalar_read_requests();
        let result = super::exl3::project(&ctx, &stream, view, x.range(0, x.bytes()).unwrap());
        let reads = ctx.scalar_read_requests() - before;
        eprintln!("OBSERVED projection 4B read_at requests={reads}, input={value}");
        if value.is_nan() {
            assert_eq!(reads, 2);
            assert!(matches!(result, Err(EngError::BadRequest(_))));
        } else {
            assert_eq!(reads, 3);
            let actual = read(&ctx, &result.unwrap());
            assert_eq!(actual.len(), 128);
            assert!(actual.iter().all(|v| v.is_finite()));
            assert!(actual.iter().all(|v| v.to_bits() == bias.to_bits()));
        }
    }
    let mut rows = [[0.0; 256]; 2];
    rows[0][0] = -1.0 / 128.0;
    rows[1][0] = 1.0 / 128.0;
    assert_rows_match_serial(&ctx, &rows, |x| {
        super::exl3::project(&ctx, &stream, view, x).unwrap()
    });
    for columns in [255, 257, 9 * 256] {
        let x = upload_f32(&ctx, &vec![0.0; columns]);
        assert!(matches!(
            super::exl3::project(&ctx, &stream, view, x.range(0, x.bytes()).unwrap()),
            Err(EngError::BadRequest(_))
        ));
    }
}

#[test]
fn exl3_scratch_matches_standalone_projection_bits() {
    if !device_present("exl3_scratch_matches_standalone_projection_bits") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let gate = weight(&ctx, [256, 128], f32::from_bits(0x3e00_1234));
    let up = weight(&ctx, [256, 128], 2.0);
    let down = weight(&ctx, [128, 256], 0.0);
    let expert = FlintExpertView {
        gate: gate.view().unwrap(),
        up: up.view().unwrap(),
        down: down.view().unwrap(),
    };
    let mut scratch = super::exl3::Scratch::new(&ctx, &FlintFfnView::Dense(expert)).unwrap();
    for view in [expert.gate, expert.up, expert.down, expert.gate] {
        let mut zero_reference = Vec::new();
        for value in [0.0, -0.0, 0.25, f32::NAN, 0.0] {
            let mut input = vec![value; view.shape[0]];
            if value == 0.25 {
                input.fill(0.0);
                input[0] = value;
                input[view.shape[0] - 1] = -0.125;
            }
            let input_buffer = upload_f32(&ctx, &input);
            let input_bytes = input_buffer.bytes();
            let x = input_buffer.range(0, input_bytes).unwrap();
            let expected = super::exl3::project(&ctx, &stream, view, x);
            let actual = scratch.project(&ctx, &stream, view, x);
            if value.is_nan() {
                assert!(matches!(expected, Err(EngError::BadRequest(_))));
                assert!(matches!(actual, Err(EngError::BadRequest(_))));
                continue;
            }
            let expected = expected.unwrap();
            let mut expected_bytes = vec![0; expected.bytes()];
            ctx.download(&expected, &mut expected_bytes).unwrap();
            if zero_reference.is_empty() {
                zero_reference = expected_bytes.clone();
            }
            if value == 0.25 {
                assert_ne!(
                    expected_bytes, zero_reference,
                    "fixture must depend on input"
                );
            }
            let actual = actual.unwrap();
            let mut actual_bytes = vec![0; actual.bytes()];
            actual
                .buffer()
                .read_at(&stream, actual.offset(), &mut actual_bytes)
                .unwrap();
            assert_eq!(actual_bytes, expected_bytes);
            let mut frame = Frame::new(&ctx, &stream).unwrap();
            let input = frame.keep(input_buffer, input_bytes);
            let published = frame.project_bf16(view, input, &mut scratch).unwrap();
            let published = read(&ctx, &frame.finish(published).unwrap());
            let expected: Vec<_> = expected_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| bf16::from_f32(f32::from_ne_bytes(*word)).to_f32())
                .collect();
            assert_eq!(bits(&published), bits(&expected));
        }
        for columns in [view.shape[0] - 1, view.shape[0] + 1, view.shape[0] * 2] {
            let x = upload_f32(&ctx, &vec![0.0; columns]);
            assert!(matches!(
                scratch.project(&ctx, &stream, view, x.range(0, x.bytes()).unwrap()),
                Err(EngError::BadRequest(_))
            ));
        }
    }
    let too_wide = weight(&ctx, [256, 512], 0.0);
    let x = upload_f32(&ctx, &[0.0; 256]);
    assert!(matches!(
        scratch.project(&ctx, &stream, too_wide.view().unwrap(), x.range(0, x.bytes()).unwrap()),
        Err(EngError::BadRequest(message)) if message.contains("exceeds its scratch slot")
    ));
}

#[test]
fn exl3_dense_ffn_publishes_bf16_without_adding_the_residual() {
    if !device_present("exl3_dense_ffn_publishes_bf16_without_adding_the_residual") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let gate = weight(&ctx, [256, 128], 1.0);
    let up = weight(&ctx, [256, 128], 2.0);
    let down = weight(&ctx, [128, 256], 0.0);
    let weights = FlintFfnView::Dense(FlintExpertView {
        gate: gate.view().unwrap(),
        up: up.view().unwrap(),
        down: down.view().unwrap(),
    });
    let mut input = [0.0; 256];
    // Constant W-prime annihilates this Hadamard column; the residual does not.
    input[1] = 0.25;
    let x = upload_f32(&ctx, &input);
    let before = ctx.scalar_read_requests();
    let output =
        super::exl3::forward(&ctx, &stream, &weights, x.range(0, x.bytes()).unwrap()).unwrap();
    let reads = ctx.scalar_read_requests() - before;
    eprintln!("OBSERVED dense FFN 4B read_at requests={reads}");
    assert_eq!(reads, 1);
    let actual = read(&ctx, &output);
    let sigmoid = bf16::from_f32(1.0 / (1.0 + (-1.0f32).exp())).to_f32();
    let hidden = bf16::from_f32(sigmoid * 2.0).to_f32();
    let decoded = f16::from_bits(Codebook::Mul1.decode_state(0)).to_f32();
    // H128 maps a constant row to its first coordinate, with factor sqrt(128).
    let expected = bf16::from_f32(decoded * 128.0 * hidden).to_f32();
    assert_eq!(actual.len(), 256);
    assert_ne!(expected, 0.0);
    for (column, value) in actual.into_iter().enumerate() {
        assert!(value.is_finite());
        assert_eq!(value, if column % 128 == 0 { expected } else { 0.0 });
    }
    for value in [f32::NAN, 0.0] {
        let x = upload_f32(&ctx, &[value; 256]);
        let result = super::exl3::forward(&ctx, &stream, &weights, x.range(0, x.bytes()).unwrap());
        if value.is_nan() {
            assert!(matches!(result, Err(EngError::ModelOutput(_))));
        } else {
            assert!(result.is_ok());
        }
    }
    let mut rows = [[0.0; 256]; 2];
    rows[0][0] = -1.0 / 128.0;
    rows[1][0] = 1.0 / 128.0;
    assert_rows_match_serial(&ctx, &rows, |x| {
        super::exl3::forward(&ctx, &stream, &weights, x).unwrap()
    });
    let x = upload_f32(&ctx, &[0.0; 9 * 256]);
    assert!(matches!(
        super::exl3::forward(&ctx, &stream, &weights, x.range(0, x.bytes()).unwrap()),
        Err(EngError::BadRequest(_))
    ));
    let malformed = FlintFfnView::Dense(FlintExpertView {
        gate: gate.view().unwrap(),
        up: down.view().unwrap(),
        down: down.view().unwrap(),
    });
    assert!(matches!(
        super::exl3::forward(&ctx, &stream, &malformed, x.range(0, 256 * 4).unwrap()),
        Err(EngError::BadRequest(_))
    ));
}

#[test]
fn exl3_moe_scales_normalized_gates_once_and_not_the_shared_expert() {
    if !device_present("exl3_moe_scales_normalized_gates_once_and_not_the_shared_expert") {
        return;
    }
    let (actual, expected, _, _) = exl3_moe_outputs([0.0; TOP_K], [0, 1, 2, 3, 4, 5], false);
    assert_eq!(bits(&actual), bits(&expected));
}

#[test]
fn exl3_moe_keeps_nonuniform_gates_paired_with_expert_ids() {
    if !device_present("exl3_moe_keeps_nonuniform_gates_paired_with_expert_ids") {
        return;
    }
    let (actual, expected, wrong_rank, _) =
        exl3_moe_outputs([-3.0, 2.0, -1.0, 3.0, -2.0, 1.0], [3, 1, 5, 2, 4, 0], false);
    assert_ne!(
        expected, wrong_rank,
        "fixture must expose wrong-rank pairing"
    );
    assert_eq!(bits(&actual), bits(&expected));
}

#[test]
fn exl3_moe_reuses_projection_scratch() {
    if !device_present("exl3_moe_reuses_projection_scratch") {
        return;
    }
    let (actual, expected, _, allocations) =
        exl3_moe_outputs([-3.0, 2.0, -1.0, 3.0, -2.0, 1.0], [3, 1, 5, 2, 4, 0], false);
    assert_eq!(bits(&actual), bits(&expected));
    // One FFN arena replaces three transients for each of the 21 projections.
    assert!(
        (1..=79).contains(&allocations),
        "expected 1..=79 successful FFN ctx.alloc calls, got {allocations}"
    );
}

#[test]
fn exl3_moe_rejects_poison_at_each_projection_position_and_recovers() {
    if !device_present("exl3_moe_rejects_poison_at_each_projection_position_and_recovers") {
        return;
    }
    let (actual, expected, _, _) =
        exl3_moe_outputs([-3.0, 2.0, -1.0, 3.0, -2.0, 1.0], [3, 1, 5, 2, 4, 0], true);
    assert_eq!(bits(&actual), bits(&expected));
}

fn exl3_moe_outputs(
    logits: [f32; TOP_K],
    rank_ids: [usize; TOP_K],
    poison_projections: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
    let mut pack = PackBuilder::tiny();
    // The existing tiny pack has four experts; this dispatch selects six.
    pack.config["num_experts"] = serde_json::json!(TOP_K);
    for layer in 1..3 {
        let prefix = format!("model.layers.{layer}.mlp");
        let router_name = format!("{prefix}.gate.weight");
        pack.plain(&router_name, "F16", &[6, 128], 0);
        for (row, logit) in pack
            .entry(&router_name)
            .bytes
            .as_chunks_mut::<{ 128 * 2 }>()
            .0
            .iter_mut()
            .zip(logits)
        {
            // The sole nonzero input is -1/128, so these weights produce exact integer logits.
            row[..2].copy_from_slice(&f16::from_f32(-128.0 * logit).to_bits().to_le_bytes());
        }
        for id in 4..TOP_K {
            for projection in ["gate", "up", "down"] {
                pack.matrix(
                    &format!("{prefix}.experts.{id}.{projection}_proj"),
                    128,
                    128,
                    4,
                );
            }
        }
        for id in 0..TOP_K {
            pack.plain(
                &format!("{prefix}.experts.{id}.up_proj.svh"),
                "F16",
                &[128],
                f16::from_f32(1.0 / 128.0).to_bits(),
            );
            pack.plain(
                &format!("{prefix}.experts.{id}.down_proj.svh"),
                "F16",
                &[128],
                f16::from_f32((id + 1) as f32 / 8.0).to_bits(),
            );
        }
    }
    let directory = pack.write();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let owner = load_flint_fixture(
        &ctx,
        &stream,
        directory.path(),
        FlintResidentLimits {
            max_staging_bytes: 16 << 20,
            max_weight_bytes: 16 << 20,
            reserve: Reserve::default(),
        },
    )
    .unwrap();
    let weights = owner.layer(1).unwrap().ffn;
    let mut input = [0.0; 128];
    input[0] = -1.0 / 128.0;
    let x = upload_f32(&ctx, &input);
    let span = x.range(0, x.bytes()).unwrap();
    let allocations_before = ctx.successful_allocations();
    let before = ctx.scalar_read_requests();
    let output = super::exl3::forward(&ctx, &stream, &weights, span).unwrap();
    let reads = ctx.scalar_read_requests() - before;
    let allocations = ctx.successful_allocations() - allocations_before;
    eprintln!("OBSERVED Tiny6 FFN 4B read_at requests={reads}, EXL3 projections=21");
    assert_eq!(
        reads, 1,
        "expected one checked FFN status read before output publication"
    );
    let actual = read(&ctx, &output);
    drop(output);
    for count in [2, 4, 8] {
        let mut rows = vec![[0.0; 128]; count];
        for (row, scale) in rows
            .iter_mut()
            .zip([-1.0, -2.0, 1.0, 2.0, -0.5, -1.5, 0.5, 1.5])
        {
            row[0] = scale / 128.0;
        }
        assert_rows_match_serial(&ctx, &rows, |x| {
            super::exl3::forward(&ctx, &stream, &weights, x).unwrap()
        });
    }
    let FlintFfnView::Moe {
        router,
        experts,
        shared,
        routing_weight_multiplier,
    } = &weights
    else {
        panic!("fixture sparse layer is not MoE");
    };
    assert_eq!(experts.len(), TOP_K);
    assert_eq!(*routing_weight_multiplier, 128.0);
    let scores = logits.map(|logit| 1.0 / (1.0 + (-logit).exp()));
    let total: f32 = scores.iter().sum();
    let gates = scores.map(|score| bf16::from_f32(score / total * 128.0).to_f32());
    let outputs: Vec<_> = (0..TOP_K)
        .map(|id| {
            let dense = FlintFfnView::Dense(experts.get(id).unwrap());
            let values = read(
                &ctx,
                &super::exl3::forward(&ctx, &stream, &dense, span).unwrap(),
            );
            assert_eq!(values.len(), 128);
            values
        })
        .collect();
    let mut expected = vec![0.0f32; 128];
    let mut wrong_rank = vec![0.0f32; 128];
    for (rank, &id) in rank_ids.iter().enumerate() {
        for (column, (sum, wrong)) in expected.iter_mut().zip(&mut wrong_rank).enumerate() {
            *sum += bf16::from_f32(outputs[id][column] * gates[id]).to_f32();
            *wrong += bf16::from_f32(outputs[rank][column] * gates[id]).to_f32();
        }
    }
    assert!(expected.iter().any(|&value| value != 0.0));
    let dense = FlintFfnView::Dense(*shared);
    let shared_output = read(
        &ctx,
        &super::exl3::forward(&ctx, &stream, &dense, span).unwrap(),
    );
    assert_eq!(shared_output.len(), expected.len());
    assert!(shared_output.iter().any(|&value| value != 0.0));
    for reference in [&mut expected, &mut wrong_rank] {
        for (sum, &shared) in reference.iter_mut().zip(&shared_output) {
            *sum = bf16::from_f32(bf16::from_f32(*sum).to_f32() + shared).to_f32();
        }
    }
    if poison_projections {
        for (name, target, is_router) in [
            (
                "first routed gate",
                experts.get(rank_ids[0]).unwrap().gate.svh,
                false,
            ),
            (
                "middle routed up",
                experts.get(rank_ids[TOP_K / 2]).unwrap().up.svh,
                false,
            ),
            (
                "last routed down",
                experts.get(rank_ids[TOP_K - 1]).unwrap().down.svh,
                false,
            ),
            ("shared down", shared.down.svh, false),
            ("router", *router, true),
        ] {
            let mut original = [0; 4];
            target
                .buffer()
                .read_at(&stream, target.offset(), &mut original)
                .unwrap();
            target
                .buffer()
                .write_at(&stream, target.offset(), &f32::NAN.to_ne_bytes())
                .unwrap();
            let before = ctx.scalar_read_requests();
            let result = super::exl3::forward(&ctx, &stream, &weights, span);
            let reads = ctx.scalar_read_requests() - before;
            target
                .buffer()
                .write_at(&stream, target.offset(), &original)
                .unwrap();
            let expected_message = if is_router {
                "status set at or before the router"
            } else {
                "late numeric/index health failed"
            };
            assert!(
                matches!(
                    (&result, is_router),
                    (Err(EngError::BadRequest(message)), true)
                        | (Err(EngError::ModelOutput(message)), false)
                        if message.contains(expected_message)
                ),
                "{name} did not fail at the expected status observation"
            );
            assert_eq!(
                reads,
                usize::from(!is_router),
                "{name} failure observation point"
            );
            let recovered = super::exl3::forward(&ctx, &stream, &weights, span).unwrap();
            let recovered = read(&ctx, &recovered);
            assert_eq!(bits(&recovered), bits(&actual), "{name} changed recovery");
            assert_eq!(
                bits(&recovered),
                bits(&expected),
                "{name} changed reference agreement"
            );
        }
    }
    (actual, expected, wrong_rank, allocations)
}

#[test]
fn exl3_scalar_read_counter_counts_successful_requests_per_thread_and_context() {
    if !device_present("exl3_scalar_read_counter_counts_successful_requests_per_thread_and_context")
    {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let buffer = ctx.upload(&[0; 8]).unwrap();
    let before = ctx.scalar_read_requests();
    buffer.read_at(&stream, 0, &mut [0; 4]).unwrap();
    assert_eq!(ctx.scalar_read_requests() - before, 1);
    buffer.read_at(&stream, 0, &mut [0; 8]).unwrap();
    assert!(buffer.read_at(&stream, 8, &mut [0; 4]).is_err());
    assert_eq!(ctx.scalar_read_requests() - before, 1);
    {
        let other = CudaContext::new().unwrap();
        let stream = other.stream().unwrap();
        let buffer = other.upload(&[0; 4]).unwrap();
        let before = other.scalar_read_requests();
        buffer.read_at(&stream, 0, &mut [0; 4]).unwrap();
        assert_eq!(other.scalar_read_requests() - before, 1);
    }
    std::thread::spawn(|| {
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let buffer = ctx.upload(&[0; 4]).unwrap();
        let before = ctx.scalar_read_requests();
        buffer.read_at(&stream, 0, &mut [0; 4]).unwrap();
        assert_eq!(ctx.scalar_read_requests() - before, 1);
    })
    .join()
    .unwrap();
    assert_eq!(ctx.scalar_read_requests() - before, 1);
}

#[test]
fn exl3_router_rejects_poison_before_consuming_valid_looking_ids() {
    if !device_present("exl3_router_rejects_poison_before_consuming_valid_looking_ids") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let router = upload_f32(&ctx, &[0.0; TOP_K * 128]);
    let block = FfnBlock::new(
        1,
        FfnShape::Moe {
            hidden: 128,
            intermediate: 128,
            shared_intermediate: 128,
            experts: TOP_K,
        },
    )
    .unwrap();
    for status in [0u32, 1, 2, 8, 1 << 31, 0] {
        let mut frame = Frame::new(&ctx, &stream).unwrap();
        let x = frame.keep(upload_f32(&ctx, &[0.0; 128]), 128 * 4);
        frame.test_set_health(status).unwrap();
        let before = ctx.scalar_read_requests();
        let result = route(
            &mut frame,
            router.range(0, router.bytes()).unwrap(),
            x,
            block,
        );
        assert_eq!(ctx.scalar_read_requests() - before, 0);
        if status != 0 {
            assert!(
                matches!(result, Err(EngError::BadRequest(message)) if message.contains("status set at or before the router")),
                "poisoned router must fail at the status observation before returning IDs"
            );
        } else {
            assert_eq!(result.unwrap().ids, [0, 1, 2, 3, 4, 5]);
        }
    }
}

#[test]
fn exl3_completion_rejects_poison_before_output_transfer_and_recovers() {
    if !device_present("exl3_completion_rejects_poison_before_output_transfer_and_recovers") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    for status in [1u32, 2, 8, 1 << 31, 0] {
        let mut frame = Frame::new(&ctx, &stream).unwrap();
        let output = frame.keep(upload_f32(&ctx, &[7.0; 128]), 128 * 4);
        frame.test_set_health(status).unwrap();
        if status != 0 {
            assert!(matches!(frame.complete(), Err(EngError::ModelOutput(_))));
            assert_eq!(frame.buffers.len(), 1);
            assert!(matches!(
                frame.finish(output),
                Err(EngError::ModelOutput(_))
            ));
        } else {
            frame.complete().unwrap();
            assert_eq!(frame.buffers.len(), 1);
            assert_eq!(read(&ctx, &frame.finish(output).unwrap()), vec![7.0; 128]);
        }
        ctx.ensure_usable().unwrap();
    }
}

#[test]
fn exl3_checked_projection_rejects_preexisting_poison_without_changing_output() {
    if !device_present("exl3_checked_projection_rejects_preexisting_poison_without_changing_output")
    {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let weight = weight(&ctx, [256, 128], 1.0);
    let x = upload_f32(&ctx, &[0.0; 256]);
    let xh = ctx.allocate(256 * 4).unwrap();
    let z = ctx.allocate(128 * 4).unwrap();
    let output = upload_f32(&ctx, &[7.0; 128]);
    for status in [1u32, 0] {
        let flag = ctx.upload(&status.to_ne_bytes()).unwrap();
        let result = weight.view().unwrap().project(
            &ctx,
            &stream,
            crate::family::project::cuda::exl3::Exl3Io {
                x: x.range(0, x.bytes()).unwrap(),
                xh: xh.range(0, xh.bytes()).unwrap(),
                z: z.range(0, z.bytes()).unwrap(),
                out: output.range(0, output.bytes()).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows: 1,
            },
        );
        if status != 0 {
            assert!(matches!(result, Err(EngError::BadRequest(_))));
            assert_eq!(read(&ctx, &output), vec![7.0; 128]);
        } else {
            result.unwrap();
            assert_eq!(read(&ctx, &output), vec![1.0; 128]);
        }
    }
}

// Reused: PackBuilder, native fixture loaders, scalar Dense FFN, and Frame::combine.
// New because: physical cached placement needs nonuniform logical experts and cold native consumers.
mod cached_tiny {
    use super::*;
    use crate::family::flint::exl3_resident::FlintExl3Resident;

    fn fixture() -> PackBuilder {
        let mut pack = PackBuilder::tiny();
        pack.config["num_experts"] = serde_json::json!(256);
        for layer in 1..3 {
            let prefix = format!("model.layers.{layer}.mlp");
            pack.plain(&format!("{prefix}.gate.weight"), "F16", &[256, 128], 0);
            for id in 0..256 {
                for (part, projection) in ["gate", "up", "down"].into_iter().enumerate() {
                    let name = format!("{prefix}.experts.{id}.{projection}_proj");
                    pack.matrix(&name, 128, 128, 4);
                    pack.entry(&format!("{name}.trellis")).bytes[0] = (id + part) as u8;
                    let suh = 0.5 + (id % 13) as f32 / 32.0;
                    let svh = match part {
                        0 => 0.25 + (id % 7) as f32 / 32.0,
                        1 => (1.0 + (id % 11) as f32 / 16.0) / 128.0,
                        _ => (id + 1) as f32 / 512.0,
                    };
                    pack.plain(
                        &format!("{name}.suh"),
                        "F16",
                        &[128],
                        f16::from_f32(suh).to_bits(),
                    );
                    pack.plain(
                        &format!("{name}.svh"),
                        "F16",
                        &[128],
                        f16::from_f32(svh).to_bits(),
                    );
                }
            }
        }
        assert!(
            pack.tensors
                .values()
                .map(|entry| entry.bytes.len())
                .sum::<usize>()
                < 64 << 20
        );
        pack
    }

    fn limits() -> FlintResidentLimits {
        FlintResidentLimits {
            max_staging_bytes: 16 << 20,
            max_weight_bytes: 64 << 20,
            reserve: Reserve::default(),
        }
    }

    fn set_routes(weights: &FlintFfnView<'_>, stream: &DeviceStream, first: usize) {
        let FlintFfnView::Moe { router, .. } = weights else {
            panic!("expected MoE")
        };
        let mut values = vec![0.0f32; 256 * 128];
        for row in 0..8 {
            for rank in 0..TOP_K {
                let id = first + (row * 4 + rank) % 32;
                values[id * 128 + row] = (rank + 1) as f32 * 64.0;
            }
        }
        let bytes: Vec<_> = values.into_iter().flat_map(f32::to_ne_bytes).collect();
        router
            .buffer()
            .write_at(stream, router.offset(), &bytes)
            .unwrap();
    }

    fn range_bytes(range: DeviceRange<'_>, stream: &DeviceStream) -> Vec<u8> {
        let mut bytes = vec![0; range.bytes()];
        range
            .buffer()
            .read_at(stream, range.offset(), &mut bytes)
            .unwrap();
        bytes
    }

    fn scalar_reference(
        ctx: &CudaContext,
        stream: &DeviceStream,
        weights: &FlintFfnView<'_>,
        rows: &[[f32; 128]],
    ) -> (Vec<f32>, Vec<i32>, Vec<f32>) {
        let FlintFfnView::Moe {
            router,
            experts,
            shared,
            routing_weight_multiplier,
        } = weights
        else {
            panic!("expected MoE")
        };
        let block = FfnBlock::new(
            rows.len(),
            FfnShape::Moe {
                hidden: 128,
                intermediate: 128,
                shared_intermediate: 128,
                experts: 256,
            },
        )
        .unwrap();
        let mut frame = Frame::new(ctx, stream).unwrap();
        let input = upload_f32(ctx, rows.as_flattened());
        let x = frame
            .input(
                input.range(0, input.bytes()).unwrap(),
                RowShape {
                    rows: rows.len(),
                    columns: 128,
                },
            )
            .unwrap();
        let routed = route(&mut frame, *router, x, block).unwrap();
        let gates = range_bytes(frame.range(routed.gates).unwrap(), stream)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_ne_bytes(*word))
            .collect::<Vec<_>>();
        let mut down = Vec::new();
        let mut shared_rows = Vec::new();
        for (row, ids) in rows.iter().zip(routed.ids.as_chunks::<TOP_K>().0) {
            let input = upload_f32(ctx, row);
            for &id in ids {
                let dense = FlintFfnView::Dense(experts.get(id as usize).unwrap());
                down.extend(read(
                    ctx,
                    &super::super::exl3::forward(
                        ctx,
                        stream,
                        &dense,
                        input.range(0, input.bytes()).unwrap(),
                    )
                    .unwrap(),
                ));
            }
            shared_rows.extend(read(
                ctx,
                &super::super::exl3::forward(
                    ctx,
                    stream,
                    &FlintFfnView::Dense(*shared),
                    input.range(0, input.bytes()).unwrap(),
                )
                .unwrap(),
            ));
        }
        let down = frame.keep(upload_f32(ctx, &down), down.len() * 4);
        let shared = frame.keep(upload_f32(ctx, &shared_rows), shared_rows.len() * 4);
        let shape = RowShape {
            rows: rows.len(),
            columns: TOP_K,
        };
        let scaled = frame.allocate(shape).unwrap();
        let scales = routing_weight_multiplier.to_ne_bytes().repeat(rows.len());
        let scale = frame.keep(ctx.upload(&scales).unwrap(), scales.len());
        ctx.row_multiply(
            stream,
            RowMulCall {
                x: frame.range(routed.gates).unwrap(),
                rhs: frame.range(scale).unwrap(),
                out: frame.range(scaled).unwrap(),
                flag: frame.flag().unwrap(),
                shape,
                rhs_columns: 1,
            },
        )
        .unwrap();
        let output = frame.combine(down, scaled, shared, block).unwrap();
        (read(ctx, &frame.finish(output).unwrap()), routed.ids, gates)
    }

    #[test]
    #[ignore = "requires explicit CUDA tiny-fixture lease and pinned-memory allowance; no absent-device skip"]
    fn cached_256_native_cold_union_matches_scalar_and_resident() {
        let directory = fixture().write();
        let ctx = CudaContext::new().expect("native tiny fixture requires actual CUDA");
        let stream = ctx.stream().unwrap();
        let resident = load_flint_fixture(&ctx, &stream, directory.path(), limits()).unwrap();
        let cached =
            FlintExl3Resident::load_cached_fixture(&ctx, &stream, directory.path(), limits())
                .unwrap();
        let source_byte = cached.test_host_cache_byte().unwrap();
        // Each tiny expert has three (8192-byte trellis + two 512-byte scales) projections.
        let stride = 3 * (8192 + 512 + 512);
        let nonrouted = resident.device_weight_bytes() - 2 * 256 * stride;
        assert_eq!(cached.device_weight_bytes(), nonrouted + 2 * 224 * stride);
        assert!(resident.device_weight_bytes() < 64 << 20);
        let off = resident.layer(1).unwrap().ffn;
        let on = cached.layer(1).unwrap().ffn;
        let FlintFfnView::Moe {
            experts: off_bank, ..
        } = &off
        else {
            panic!("expected MoE")
        };
        let FlintFfnView::Moe {
            experts: on_bank, ..
        } = &on
        else {
            panic!("expected MoE")
        };
        assert_eq!(on_bank.shape(0).unwrap(), (128, 128));
        assert!(on_bank.get(0).is_err(), "metadata must not prepare payload");
        let mut rows = [[0.0; 128]; 8];
        for (index, row) in rows.iter_mut().enumerate() {
            row[index] = (index + 1) as f32 / 128.0;
            row[16 + index] = -1.0 / 256.0;
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut previous = None;
        for first in [224, 32, 64, 96, 128, 160, 192, 224] {
            set_routes(&off, &stream, first);
            set_routes(&on, &stream, first);
            let (expected, ids, gates) = scalar_reference(&ctx, &stream, &off, &rows);
            assert!(expected.iter().chain(&gates).all(|value| value.is_finite()));
            assert_ne!(gates[0], gates[TOP_K - 1], "gates must vary by rank");
            let union = ids
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(ids.len(), 48);
            assert_eq!(
                union,
                (first as i32..first as i32 + 32).collect::<std::collections::BTreeSet<_>>()
            );
            assert!(
                union.len() < ids.len(),
                "raw routing must retain duplicates"
            );
            seen.extend(union);
            let input = upload_f32(&ctx, rows.as_flattened());
            let span = input.range(0, input.bytes()).unwrap();
            let actual = super::super::exl3::forward(&ctx, &stream, &on, span).unwrap();
            assert_eq!(bits(&read(&ctx, &actual)), bits(&expected));
            assert!(
                matches!(cached.test_host_cache_byte(), Err(EngError::BadRequest(message)) if message.contains("copy outstanding")),
                "Frame completion must not clear pinned transfer bookkeeping"
            );
            let off_output = super::super::exl3::forward(&ctx, &stream, &off, span).unwrap();
            assert_eq!(bits(&read(&ctx, &off_output)), bits(&expected));
            assert_ne!(
                bits(&expected[..128]),
                bits(&expected[128..256]),
                "rows must differ"
            );
            if let Some(previous) = &previous {
                assert_ne!(
                    &bits(&expected),
                    previous,
                    "routing and payload must affect output"
                );
            }
            previous = Some(bits(&expected));
            for row in &rows {
                let (reference, _, _) =
                    scalar_reference(&ctx, &stream, &off, std::slice::from_ref(row));
                let input = upload_f32(&ctx, row);
                let span = input.range(0, input.bytes()).unwrap();
                let result = super::super::exl3::forward(&ctx, &stream, &on, span).unwrap();
                assert_eq!(bits(&read(&ctx, &result)), bits(&reference));
                let result = super::super::exl3::forward(&ctx, &stream, &off, span).unwrap();
                assert_eq!(bits(&read(&ctx, &result)), bits(&reference));
            }
            // Compare every converted component after native slot relocation, not only final logits.
            on_bank
                .with_prepared(&ctx, &stream, &ids, |prepared| {
                    for &id in &ids {
                        let actual = prepared.get(id as usize)?;
                        let expected = off_bank.get(id as usize)?;
                        for (actual, expected) in [actual.gate, actual.up, actual.down]
                            .into_iter()
                            .zip([expected.gate, expected.up, expected.down])
                        {
                            for (actual, expected) in [actual.trellis, actual.suh, actual.svh]
                                .into_iter()
                                .zip([expected.trellis, expected.suh, expected.svh])
                            {
                                assert_eq!(
                                    range_bytes(actual, &stream),
                                    range_bytes(expected, &stream)
                                );
                            }
                        }
                    }
                    super::super::exl3::forward(
                        &ctx,
                        &stream,
                        &FlintFfnView::Dense(prepared.get(ids[0] as usize)?),
                        span,
                    )
                })
                .unwrap();
        }
        assert_eq!(seen, (32..256).collect::<std::collections::BTreeSet<_>>());
        // A real completion-health refusal must poison the model until drained canonical recovery.
        let failure =
            on_bank.with_prepared(&ctx, &stream, &[224, 225, 226, 227, 228, 229], |prepared| {
                let input = upload_f32(&ctx, &rows[0]);
                let output = super::super::exl3::forward(
                    &ctx,
                    &stream,
                    &FlintFfnView::Dense(prepared.get(224)?),
                    input.range(0, input.bytes())?,
                )?;
                let mut frame = Frame::new(&ctx, &stream)?;
                let bytes = output.bytes();
                let output = frame.keep(output, bytes);
                frame.test_set_health(8)?;
                frame.finish(output)
            });
        assert!(matches!(failure, Err(EngError::ModelOutput(_))));
        let input = upload_f32(&ctx, rows.as_flattened());
        assert!(
            matches!(super::super::exl3::forward(&ctx, &stream, &on, input.range(0, input.bytes()).unwrap()), Err(EngError::BadRequest(message)) if message.contains("poisoned"))
        );
        stream.synchronize().unwrap();
        cached.recover_expert_cache(&ctx, &stream).unwrap();
        assert_eq!(cached.test_host_cache_byte().unwrap(), source_byte);
        let (expected, _, _) = scalar_reference(&ctx, &stream, &off, &rows);
        let recovered =
            super::super::exl3::forward(&ctx, &stream, &on, input.range(0, input.bytes()).unwrap())
                .unwrap();
        assert_eq!(bits(&read(&ctx, &recovered)), bits(&expected));
        cached.recover_expert_cache(&ctx, &stream).unwrap();
    }
}
