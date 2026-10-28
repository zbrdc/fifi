//! Positive NVFP4 fixtures use FP16-exact weights; fault cases leave that domain.
//! Approximate/non-FP16-exact accumulation and model quality need a separate gate;
//! the scalar BF16 graph deliberately does not imitate the backend's FP16 losses.

mod first_light;
pub(super) mod profile;
mod real_exl3;
mod real_gguf;
mod s2_nvfp4_dense;
mod s3_nvfp4_ffn;
pub(in crate::family::flint::cuda_ffn) use real_gguf::decode;

use super::super::reference::{self, Expert, Fixture, Kind, Matrix};
use super::*;
use crate::kernels::cuda::device_present;

const H: usize = 32;

#[derive(Clone, Copy)]
struct Diagonal {
    code: u8,
    scale: u8,
    decoded: f32,
    factor: f32,
}

const UNIT: Diagonal = Diagonal {
    code: 2,
    scale: 0x38,
    decoded: 1.0,
    factor: 1.0,
};

fn assert_normal_fp16_operand(value: f32) {
    // A normal half retains ten of F32's 23 fraction bits; exclude subnormal calibration.
    assert!(
        value == 0.0
            || ((1.0 / 16384.0..=65504.0).contains(&value.abs()) && value.to_bits() & 0x1fff == 0),
        "fixture weight is not a normal FP16-exact operand: {value}"
    );
}

struct Weight {
    packed: DeviceBuffer,
    scales: DeviceBuffer,
    cpu: Matrix,
}

impl Weight {
    fn new(ctx: &CudaContext, shape: [usize; 2], spec: Diagonal) -> Self {
        assert_normal_fp16_operand(spec.decoded * spec.factor);
        let [rows, columns] = shape;
        let mut bytes = vec![0; rows * columns / 2];
        let mut values = vec![0.0; rows * columns];
        for row in 0..rows {
            let index = row * columns + row % columns;
            bytes[index / 2] |= spec.code << ((index % 2) * 4);
            values[index] = spec.decoded;
        }
        Self {
            packed: ctx.upload(&bytes).unwrap(),
            scales: ctx.upload(&vec![spec.scale; rows * columns / 16]).unwrap(),
            cpu: Matrix {
                rows,
                cols: columns,
                values,
                factor: spec.factor,
            },
        }
    }

    fn view<'a>(&'a self, ctx: &CudaContext) -> Nvfp4WeightRange<'a> {
        ctx.nvfp4_weight_range(
            self.packed.range(0, self.packed.bytes()).unwrap(),
            self.scales.range(0, self.scales.bytes()).unwrap(),
            self.cpu.factor,
            [self.cpu.rows, self.cpu.cols],
        )
        .unwrap()
    }
}

struct OwnedExpert {
    gate: Weight,
    up: Weight,
    down: Weight,
}

impl OwnedExpert {
    fn new(ctx: &CudaContext, intermediate: usize, specs: [Diagonal; 3]) -> Self {
        Self {
            gate: Weight::new(ctx, [intermediate, H], specs[0]),
            up: Weight::new(ctx, [intermediate, H], specs[1]),
            down: Weight::new(ctx, [H, intermediate], specs[2]),
        }
    }

    fn view<'a>(&'a self, ctx: &CudaContext) -> ExpertWeights<'a> {
        ExpertWeights {
            gate: self.gate.view(ctx),
            up: self.up.view(ctx),
            down: self.down.view(ctx),
        }
    }

    fn cpu(&self) -> Expert {
        Expert {
            gate: self.gate.cpu.clone(),
            up: self.up.cpu.clone(),
            down: self.down.cpu.clone(),
        }
    }
}

struct Rig {
    dense: OwnedExpert,
    shared: OwnedExpert,
    experts: Vec<OwnedExpert>,
    router: DeviceBuffer,
    router_cpu: Matrix,
}

impl Rig {
    fn new(ctx: &CudaContext, wide: bool) -> Self {
        let gate = if wide {
            Diagonal {
                code: 4,
                scale: 0x70,
                decoded: 256.0,
                factor: 1.0,
            }
        } else {
            UNIT
        };
        let up = if wide {
            Diagonal {
                code: 4,
                scale: 0x78,
                decoded: 512.0,
                factor: 1.0,
            }
        } else {
            Diagonal {
                factor: 0.5,
                ..UNIT
            }
        };
        let specs = [gate, up, UNIT];
        let mut router = vec![0.0f32; 8 * H];
        // All eight F32 sigmoid scores saturate to 1, despite unequal raw logits.
        for expert in 0..8 {
            router[expert * H] = 100.0 + expert as f32;
        }
        Self {
            dense: OwnedExpert::new(ctx, 64, specs),
            shared: OwnedExpert::new(
                ctx,
                96,
                [
                    gate,
                    up,
                    Diagonal {
                        factor: 0.25,
                        ..UNIT
                    },
                ],
            ),
            experts: (0..8)
                .map(|e| {
                    OwnedExpert::new(
                        ctx,
                        64,
                        [
                            gate,
                            up,
                            Diagonal {
                                factor: 1.0 + e as f32 / 8.0,
                                ..UNIT
                            },
                        ],
                    )
                })
                .collect(),
            router: ctx.upload(&f32_bytes(&router)).unwrap(),
            router_cpu: Matrix {
                rows: 8,
                cols: H,
                values: router,
                factor: 1.0,
            },
        }
    }

    fn cpu(&self) -> Fixture {
        Fixture {
            hidden: H,
            router: self.router_cpu.clone(),
            experts: self.experts.iter().map(OwnedExpert::cpu).collect(),
            shared: self.shared.cpu(),
            dense: self.dense.cpu(),
        }
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn input(rows: usize, wide: bool) -> Vec<f32> {
    (0..rows * H)
        .map(|i| {
            if wide || i % H == 0 {
                1.0
            } else {
                0.5 + ((i / H + i % H) % 7) as f32 / 8.0
            }
        })
        .collect()
}

fn assert_rows(buffer: &DeviceBuffer, stream: &DeviceStream, expected: &[f32]) {
    let mut bytes = vec![0; expected.len() * 4];
    buffer.read_at(stream, 0, &mut bytes).unwrap();
    for (index, (bytes, expected)) in bytes.as_chunks::<4>().0.iter().zip(expected).enumerate() {
        let actual = f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(
            actual.to_bits() & 0xffff,
            0,
            "unpublished BF16 row element {index}"
        );
        assert_eq!(
            actual, *expected,
            "reference mismatch at row element {index}"
        );
    }
}

fn forty_expert_rig(ctx: &CudaContext) -> Rig {
    let mut rig = Rig::new(ctx, false);
    rig.experts = (0..40)
        .map(|expert| {
            OwnedExpert::new(
                ctx,
                64,
                [
                    UNIT,
                    Diagonal {
                        factor: 0.5,
                        ..UNIT
                    },
                    Diagonal {
                        factor: 1.0 + expert as f32 / 64.0,
                        ..UNIT
                    },
                ],
            )
        })
        .collect();
    rig.router_cpu.rows = 40;
    rig.router_cpu.values = vec![0.0; 40 * H];
    for expert in 0..40 {
        rig.router_cpu.values[expert * H] = (expert as f32 - 19.5) / 32.0;
    }
    rig.router = ctx.upload(&f32_bytes(&rig.router_cpu.values)).unwrap();
    rig
}

fn run_moe_test(
    ctx: &CudaContext,
    stream: &DeviceStream,
    rig: &Rig,
    x: &[f32],
) -> Result<DeviceBuffer, EngError> {
    assert!(x.len().is_multiple_of(H));
    let bank: Vec<_> = rig.experts.iter().map(|expert| expert.view(ctx)).collect();
    let weights = FfnWeights::Moe {
        router: rig
            .router
            .range(0, rig.router_cpu.values.len() * 4)
            .unwrap(),
        experts: &bank,
        shared: rig.shared.view(ctx),
    };
    let source = ctx.upload(&f32_bytes(x)).unwrap();
    forward(
        ctx,
        stream,
        FfnCall {
            weights: &weights,
            x: source.range(0, x.len() * 4).unwrap(),
            rows: x.len() / H,
        },
    )
}

#[test]
fn trow_moe_varies_routes_across_tokens_and_skips_unused_experts() {
    if !device_present("trow_moe_varies_routes_across_tokens_and_skips_unused_experts") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let rig = forty_expert_rig(&ctx);
    let mut x = input(17, false);
    for token in 0..17 {
        x[token * H] = if token % 2 == 0 { 1.0 } else { -1.0 };
    }
    let expected = reference::run(&rig.cpu(), &x, 17, Kind::Moe).unwrap();
    for (token, ids) in expected.ids.as_chunks::<6>().0.iter().enumerate() {
        assert_eq!(
            ids,
            if token % 2 == 0 {
                &[39, 38, 37, 36, 35, 34]
            } else {
                &[0, 1, 2, 3, 4, 5]
            }
        );
    }
    assert!(expected.ids.iter().all(|id| !(6..34).contains(id)));
    let output = run_moe_test(&ctx, &stream, &rig, &x).unwrap();
    assert_rows(&output, &stream, &expected.output);
}

#[test]
fn trow_router_many_expert_ties_refuses_zero_mass_and_recovers() {
    if !device_present("trow_router_many_expert_ties_refuses_zero_mass_and_recovers") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rig = forty_expert_rig(&ctx);
    for expert in 0..40 {
        rig.router_cpu.values[expert * H] = 100.0 + expert as f32;
    }
    rig.router = ctx.upload(&f32_bytes(&rig.router_cpu.values)).unwrap();
    let x = input(17, true);
    let expected = reference::run(&rig.cpu(), &x, 17, Kind::Moe).unwrap();
    assert!(
        expected
            .ids
            .as_chunks::<6>()
            .0
            .iter()
            .all(|&ids| ids == [0, 1, 2, 3, 4, 5])
    );
    let output = run_moe_test(&ctx, &stream, &rig, &x).unwrap();
    assert_rows(&output, &stream, &expected.output);
    let valid_router = rig.router_cpu.values.clone();
    for expert in 0..40 {
        rig.router_cpu.values[expert * H] = -1000.0;
    }
    rig.router = ctx.upload(&f32_bytes(&rig.router_cpu.values)).unwrap();
    assert!(reference::run(&rig.cpu(), &x, 17, Kind::Moe).is_err());
    let before = ctx.scalar_read_requests();
    let error = run_moe_test(&ctx, &stream, &rig, &x)
        .err()
        .expect("zero selected mass");
    assert_eq!(ctx.scalar_read_requests() - before, 0);
    assert!(
        matches!(&error, EngError::BadRequest(message) if message.contains("status set at or before the router")),
        "{error:?}"
    );
    rig.router_cpu.values = valid_router;
    rig.router = ctx.upload(&f32_bytes(&rig.router_cpu.values)).unwrap();
    let retry = run_moe_test(&ctx, &stream, &rig, &x).unwrap();
    assert_rows(&retry, &stream, &expected.output);
}

#[test]
fn trow_dense_projection_accumulates_signed_intermediate_tail_above_2048() {
    if !device_present("trow_dense_projection_accumulates_signed_intermediate_tail_above_2048") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rig = Rig::new(&ctx, false);
    rig.dense = OwnedExpert::new(
        &ctx,
        2080,
        [
            UNIT,
            Diagonal {
                factor: 0.5,
                ..UNIT
            },
            Diagonal {
                factor: 1.25,
                ..UNIT
            },
        ],
    );
    let mut packed = vec![0u8; H * 2080 / 2];
    rig.dense.down.cpu.values.fill(0.0);
    for row in 0..H {
        for (column, code, decoded) in [(row, 2, 1.0), (2048 + (row + 1) % H, 9, -0.5)] {
            assert_normal_fp16_operand(decoded * rig.dense.down.cpu.factor);
            let index = row * 2080 + column;
            packed[index / 2] |= code << ((index % 2) * 4);
            rig.dense.down.cpu.values[index] = decoded;
        }
    }
    rig.dense.down.packed = ctx.upload(&packed).unwrap();
    let x = input(17, false);
    let expected = reference::run(&rig.cpu(), &x, 17, Kind::Dense).unwrap();
    assert!(expected.output.iter().any(|&value| value != 0.0));
    let source = ctx.upload(&f32_bytes(&x)).unwrap();
    let weights = FfnWeights::Dense(rig.dense.view(&ctx));
    let output = forward(
        &ctx,
        &stream,
        FfnCall {
            weights: &weights,
            x: source.range(0, x.len() * 4).unwrap(),
            rows: 17,
        },
    )
    .unwrap();
    assert_rows(&output, &stream, &expected.output);
}

#[test]
fn nvfp4_operand_overflow_is_checked_before_publication_and_a_retry_survives() {
    if !device_present("nvfp4_operand_overflow_is_checked_before_publication_and_a_retry_survives")
    {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let rig = Rig::new(&ctx, false);
    let mut bad = OwnedExpert::new(&ctx, 64, [UNIT; 3]);
    // Deliberately leave the calibrated fixture domain to exercise the device ticket.
    bad.down.cpu.factor = f32::MAX;
    let x = input(7, true);
    let source = ctx.upload(&f32_bytes(&x)).unwrap();
    let bad_weights = FfnWeights::Dense(bad.view(&ctx));
    let error = forward(
        &ctx,
        &stream,
        FfnCall {
            weights: &bad_weights,
            x: source.range(0, x.len() * 4).unwrap(),
            rows: 7,
        },
    )
    .err()
    .expect("invalid FP16 weight operand");
    assert!(
        format!("{error:?}").contains("FP16 operand overflow"),
        "{error:?}"
    );
    let good = FfnWeights::Dense(rig.dense.view(&ctx));
    let expected = reference::run(&rig.cpu(), &x, 7, Kind::Dense).unwrap();
    let retry = forward(
        &ctx,
        &stream,
        FfnCall {
            weights: &good,
            x: source.range(0, x.len() * 4).unwrap(),
            rows: 7,
        },
    )
    .unwrap();
    assert_rows(&retry, &stream, &expected.output);
}

#[test]
fn trow_moe_matches_cpu_bf16_graph_for_all_t_with_saturated_score_ties() {
    if !device_present("trow_moe_matches_cpu_bf16_graph_for_all_t_with_saturated_score_ties") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let rig = Rig::new(&ctx, false);
    let fixture = rig.cpu();
    let bank: Vec<_> = rig.experts.iter().map(|e| e.view(&ctx)).collect();
    let weights = FfnWeights::Moe {
        router: rig.router.range(0, 8 * H * 4).unwrap(),
        experts: &bank,
        shared: rig.shared.view(&ctx),
    };
    for rows in [1, 7, 17, 512] {
        let x = input(rows, false);
        let expected = reference::run(&fixture, &x, rows, Kind::Moe).unwrap();
        assert!(
            expected
                .ids
                .as_chunks::<6>()
                .0
                .iter()
                .all(|&ids| ids == [0, 1, 2, 3, 4, 5])
        );
        let source = ctx.upload(&f32_bytes(&x)).unwrap();
        let output = forward(
            &ctx,
            &stream,
            FfnCall {
                weights: &weights,
                x: source.range(0, x.len() * 4).unwrap(),
                rows,
            },
        )
        .unwrap();
        assert_rows(&output, &stream, &expected.output);
    }
}

#[test]
fn trow_router_keeps_f32_scores_until_selected_normalization() {
    if !device_present("trow_router_keeps_f32_scores_until_selected_normalization") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rig = Rig::new(&ctx, false);
    for expert in 0..8 {
        rig.router_cpu.values[expert * H] = expert as f32 / 256.0;
    }
    rig.router = ctx.upload(&f32_bytes(&rig.router_cpu.values)).unwrap();
    let fixture = rig.cpu();
    let bank: Vec<_> = rig.experts.iter().map(|e| e.view(&ctx)).collect();
    let weights = FfnWeights::Moe {
        router: rig.router.range(0, 8 * H * 4).unwrap(),
        experts: &bank,
        shared: rig.shared.view(&ctx),
    };
    for rows in [1, 17] {
        let x = input(rows, false);
        let expected = reference::run(&fixture, &x, rows, Kind::Moe).unwrap();
        assert!(
            expected
                .ids
                .as_chunks::<6>()
                .0
                .iter()
                .all(|&ids| ids == [7, 6, 5, 4, 3, 2])
        );
        let source = ctx.upload(&f32_bytes(&x)).unwrap();
        let output = forward(
            &ctx,
            &stream,
            FfnCall {
                weights: &weights,
                x: source.range(0, x.len() * 4).unwrap(),
                rows,
            },
        )
        .unwrap();
        assert_rows(&output, &stream, &expected.output);
    }
}

#[test]
fn trow_unselected_raw_infinite_logit_is_not_hidden_by_sigmoid() {
    if !device_present("trow_unselected_raw_infinite_logit_is_not_hidden_by_sigmoid") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rig = Rig::new(&ctx, false);
    rig.router_cpu.values.fill(0.0);
    // Negative overflow would score zero and stay unselected if sigmoid hid it.
    rig.router_cpu.values[7 * H] = f32::from_bits(0xff7f0000);
    rig.router = ctx.upload(&f32_bytes(&rig.router_cpu.values)).unwrap();
    let bank: Vec<_> = rig.experts.iter().map(|e| e.view(&ctx)).collect();
    let weights = FfnWeights::Moe {
        router: rig.router.range(0, 8 * H * 4).unwrap(),
        experts: &bank,
        shared: rig.shared.view(&ctx),
    };
    let mut x = input(7, true);
    for row in x.as_chunks_mut::<H>().0 {
        row[0] = 2.0;
    }
    assert!(reference::run(&rig.cpu(), &x, 7, Kind::Moe).is_err());
    let source = ctx.upload(&f32_bytes(&x)).unwrap();
    let before = ctx.scalar_read_requests();
    let error = forward(
        &ctx,
        &stream,
        FfnCall {
            weights: &weights,
            x: source.range(0, x.len() * 4).unwrap(),
            rows: 7,
        },
    )
    .err()
    .expect("ALL raw logits must be healthy");
    assert_eq!(ctx.scalar_read_requests() - before, 0);
    assert!(
        matches!(&error, EngError::BadRequest(message) if message.contains("status set at or before the router")),
        "{error:?}"
    );
}

#[test]
fn trow_dense_prefix_layers_use_the_same_block_path_without_routing() {
    if !device_present("trow_dense_prefix_layers_use_the_same_block_path_without_routing") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let rig = Rig::new(&ctx, false);
    for prefix in 0..2 {
        let weight = OwnedExpert::new(
            &ctx,
            64,
            [
                UNIT,
                UNIT,
                Diagonal {
                    factor: 0.5 + prefix as f32 * 0.25,
                    ..UNIT
                },
            ],
        );
        let mut fixture = rig.cpu();
        fixture.dense = weight.cpu();
        let weights = FfnWeights::Dense(weight.view(&ctx));
        for rows in [1, 7, 17, 512] {
            let x = input(rows, false);
            let expected = reference::run(&fixture, &x, rows, Kind::Dense).unwrap();
            let source = ctx.upload(&f32_bytes(&x)).unwrap();
            let output = forward(
                &ctx,
                &stream,
                FfnCall {
                    weights: &weights,
                    x: source.range(0, x.len() * 4).unwrap(),
                    rows,
                },
            )
            .unwrap();
            assert_rows(&output, &stream, &expected.output);
        }
    }
}

#[test]
fn trow_silu_keeps_the_sigmoid_bf16_endpoint_before_multiplication() {
    if !device_present("trow_silu_keeps_the_sigmoid_bf16_endpoint_before_multiplication") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let rig = Rig::new(&ctx, false);
    let weight = OwnedExpert::new(
        &ctx,
        64,
        [
            Diagonal {
                code: 10,
                scale: 0x38,
                decoded: -1.0,
                factor: 3.25,
            },
            UNIT,
            UNIT,
        ],
    );
    let mut fixture = rig.cpu();
    fixture.dense = weight.cpu();
    let weights = FfnWeights::Dense(weight.view(&ctx));
    let x = input(17, true);
    let expected = reference::run(&fixture, &x, 17, Kind::Dense).unwrap();
    let source = ctx.upload(&f32_bytes(&x)).unwrap();
    let output = forward(
        &ctx,
        &stream,
        FfnCall {
            weights: &weights,
            x: source.range(0, x.len() * 4).unwrap(),
            rows: 17,
        },
    )
    .unwrap();
    assert_rows(&output, &stream, &expected.output);
}

#[test]
fn trow_wide_hidden_and_restored_down_never_round_through_f16() {
    if !device_present("trow_wide_hidden_and_restored_down_never_round_through_f16") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let rig = Rig::new(&ctx, true);
    let fixture = rig.cpu();
    let bank: Vec<_> = rig.experts.iter().map(|e| e.view(&ctx)).collect();
    let moe = FfnWeights::Moe {
        router: rig.router.range(0, 8 * H * 4).unwrap(),
        experts: &bank,
        shared: rig.shared.view(&ctx),
    };
    let dense = FfnWeights::Dense(rig.dense.view(&ctx));
    for (kind, weights) in [(Kind::Moe, &moe), (Kind::Dense, &dense)] {
        for rows in [1, 7, 17, 512] {
            let x = input(rows, true);
            let expected = reference::run(&fixture, &x, rows, kind).unwrap();
            assert!(expected.max_intermediate > 65504.0);
            assert!(expected.output.iter().any(|v| v.abs() > 65504.0));
            let source = ctx.upload(&f32_bytes(&x)).unwrap();
            let output = forward(
                &ctx,
                &stream,
                FfnCall {
                    weights,
                    x: source.range(0, x.len() * 4).unwrap(),
                    rows,
                },
            )
            .unwrap();
            assert_rows(&output, &stream, &expected.output);
        }
    }
}

#[test]
fn late_bf16_overflow_refuses_finite_projection_and_a_valid_retry_survives() {
    if !device_present("late_bf16_overflow_refuses_finite_projection_and_a_valid_retry_survives") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let rig = Rig::new(&ctx, false);
    let bad = OwnedExpert::new(
        &ctx,
        64,
        [
            UNIT,
            UNIT,
            Diagonal {
                factor: 1023.0 / 256.0,
                ..UNIT
            },
        ],
    );
    // Hidden=2^126, raw down=4092: legal FP16 operands, finite undo, BF16 overflow.
    let projected = 2.0_f32.powi(126) * (1023.0 / 256.0);
    assert!(projected.is_finite());
    assert_eq!(projected.to_bits(), 0x7f7f_c000);
    let mut fixture = rig.cpu();
    fixture.dense = bad.cpu();
    let x = [2.0_f32.powi(63); 7 * H];
    assert!(
        reference::run(&fixture, &x, 7, Kind::Dense)
            .unwrap_err()
            .contains("BF16 publication overflow")
    );
    let source = ctx.upload(&f32_bytes(&x)).unwrap();
    let bad_weights = FfnWeights::Dense(bad.view(&ctx));
    let error = forward(
        &ctx,
        &stream,
        FfnCall {
            weights: &bad_weights,
            x: source.range(0, x.len() * 4).unwrap(),
            rows: 7,
        },
    )
    .err()
    .expect("finite F32 must still fail BF16 overflow");
    assert!(
        matches!(&error, EngError::ModelOutput(message) if message.contains("late numeric/index health")),
        "{error:?}"
    );
    let retry_x = input(7, true);
    let retry_source = ctx.upload(&f32_bytes(&retry_x)).unwrap();
    let good = FfnWeights::Dense(rig.dense.view(&ctx));
    let expected = reference::run(&rig.cpu(), &retry_x, 7, Kind::Dense).unwrap();
    let output = forward(
        &ctx,
        &stream,
        FfnCall {
            weights: &good,
            x: retry_source.range(0, retry_x.len() * 4).unwrap(),
            rows: 7,
        },
    )
    .unwrap();
    assert_rows(&output, &stream, &expected.output);
}

#[test]
fn shared_row_ops_keep_f32_math_and_sticky_numeric_bound_health() {
    if !device_present("shared_row_ops_keep_f32_math_and_sticky_numeric_bound_health") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let shape = RowShape {
        rows: 1,
        columns: 4,
    };
    let flag = ctx.upload(&[0; 4]).unwrap();
    let source = ctx.upload(&f32_bytes(&[-3.25; 4])).unwrap();
    let out = ctx.allocate(16).unwrap();
    ctx.row_sigmoid(
        &stream,
        RowUnaryCall {
            x: source.range(0, 16).unwrap(),
            out: out.range(0, 16).unwrap(),
            flag: flag.range(0, 4).unwrap(),
            shape,
        },
    )
    .unwrap();
    let mut bytes = [0; 16];
    out.read_at(&stream, 0, &mut bytes).unwrap();
    let actual = f32::from_ne_bytes(bytes[..4].try_into().unwrap());
    let expected = 1.0 / (1.0 + 3.25f32.exp());
    assert!((actual - expected).abs() < 1e-7);
    assert_ne!(
        actual.to_bits() & 0xffff,
        0,
        "shared sigmoid must not BF16-publish"
    );

    let nan = ctx.upload(&f32_bytes(&[f32::NAN; 4])).unwrap();
    ctx.row_sigmoid(
        &stream,
        RowUnaryCall {
            x: nan.range(0, 16).unwrap(),
            out: out.range(0, 16).unwrap(),
            flag: flag.range(0, 4).unwrap(),
            shape,
        },
    )
    .unwrap();
    assert_rows(&out, &stream, &[0.0; 4]);
    let max = ctx.upload(&f32_bytes(&[f32::MAX; 8])).unwrap();
    let twice = ctx.upload(&f32_bytes(&[2.0])).unwrap();
    ctx.row_multiply(
        &stream,
        RowMulCall {
            x: max.range(0, 16).unwrap(),
            rhs: twice.range(0, 4).unwrap(),
            out: out.range(0, 16).unwrap(),
            flag: flag.range(0, 4).unwrap(),
            shape,
            rhs_columns: 1,
        },
    )
    .unwrap();
    assert_rows(&out, &stream, &[0.0; 4]);
    ctx.row_rank_sum(
        &stream,
        RowSumCall {
            x: max.range(0, 32).unwrap(),
            out: out.range(0, 16).unwrap(),
            flag: flag.range(0, 4).unwrap(),
            shape,
            rank: 2,
        },
    )
    .unwrap();
    assert_rows(&out, &stream, &[0.0; 4]);
    let rows = ctx
        .upload(&f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
        .unwrap();
    let ids: Vec<_> = [-1i32, 2, 1].iter().flat_map(|v| v.to_ne_bytes()).collect();
    let ids = ctx.upload(&ids).unwrap();
    let gathered = ctx.allocate(48).unwrap();
    ctx.row_gather(
        &stream,
        RowGatherCall {
            x: rows.range(0, 32).unwrap(),
            indices: ids.range(0, 12).unwrap(),
            out: gathered.range(0, 48).unwrap(),
            flag: flag.range(0, 4).unwrap(),
            shape: RowShape {
                rows: 3,
                columns: 4,
            },
            source_rows: 2,
        },
    )
    .unwrap();
    assert_rows(
        &gathered,
        &stream,
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0],
    );
    let mut health = [0; 4];
    flag.read_at(&stream, 0, &mut health).unwrap();
    assert_eq!(
        u32::from_ne_bytes(health),
        1 | 8,
        "numeric failures and bad IDs must both remain sticky"
    );
}
