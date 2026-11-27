//! Device tests of the fused controller chain at the wrapper boundary: the two launches give the
//! bits of the separate launches they replace (gate norms, the down reduction, the paired
//! projections and the controller norm), report a non-finite role the same way, and refuse a
//! row that overlaps another before any launch.

use super::*;
use crate::kernels::cuda::{
    DeviceBuffer, DeviceStream,
    resident::dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape},
    resident::inkling_stream::NormCall,
    testkit::{self, Rng},
};

const GATE_EPS: f32 = 1e-5;
const EPS: f32 = 1e-5;

/// The rows of one role, one buffer each, and its weights.
struct Role {
    input: DeviceBuffer,
    message: DeviceBuffer,
    gamma: DeviceBuffer,
    gate: DeviceBuffer,
    reduced: DeviceBuffer,
    gain: DeviceBuffer,
    forget: DeviceBuffer,
    out: DeviceBuffer,
    health: DeviceBuffer,
    gate_health: DeviceBuffer,
    down: DeviceBuffer,
    gain_w: DeviceBuffer,
    forget_w: DeviceBuffer,
}

#[derive(Clone, Copy)]
struct Shape {
    width: usize,
    rank: usize,
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect()
}

fn upload(ctx: &CudaContext, stream: &DeviceStream, values: &[f32]) -> DeviceBuffer {
    let buffer = ctx.allocate(values.len() * WORD).expect("allocate");
    buffer.write_at(stream, 0, &bytes(values)).expect("write");
    buffer
}

fn zeros(ctx: &CudaContext, words: usize) -> DeviceBuffer {
    ctx.allocate(words * WORD).expect("allocate")
}

impl Role {
    /// Random rows and weights: the inputs are near unit scale, the weights small, so every
    /// value stays finite; `broken` plants a NaN in the message of the role.
    fn new(
        ctx: &CudaContext,
        stream: &DeviceStream,
        rng: &mut Rng,
        Shape { width, rank }: Shape,
        broken: bool,
    ) -> Self {
        let row = |rng: &mut Rng, n: usize, sd: f64| {
            (0..n)
                .map(|_| (rng.normal() * sd) as f32)
                .collect::<Vec<_>>()
        };
        let mut message = row(rng, width, 1.0);
        if broken {
            message[width / 2] = f32::NAN;
        }
        Self {
            input: upload(ctx, stream, &row(rng, width, 1.0)),
            message: upload(ctx, stream, &message),
            gamma: upload(ctx, stream, &row(rng, width, 0.5)),
            gate: zeros(ctx, 2 * width),
            reduced: zeros(ctx, rank),
            gain: zeros(ctx, width),
            forget: zeros(ctx, width),
            out: zeros(ctx, width),
            health: zeros(ctx, 1),
            gate_health: zeros(ctx, 1),
            down: upload(ctx, stream, &row(rng, rank * 2 * width, 0.02)),
            gain_w: upload(ctx, stream, &row(rng, width * rank, 0.1)),
            forget_w: upload(ctx, stream, &row(rng, width * rank, 0.1)),
        }
    }

    fn gemv<'b>(
        &'b self,
        (w, x, y): (&'b DeviceBuffer, &'b DeviceBuffer, &'b DeviceBuffer),
        [rows, cols]: [usize; 2],
    ) -> DenseGemvCall<'b> {
        DenseGemvCall {
            shape: DenseGemvShape::packed(1, rows, cols),
            dtype: DenseDtype::F32,
            w: w.range(0, rows * cols * WORD).unwrap(),
            x: x.range(0, cols * WORD).unwrap(),
            y: y.range(0, rows * WORD).unwrap(),
        }
    }
}

fn norm_row<'a>(
    role: &'a Role,
    row: usize,
    src: &'a DeviceBuffer,
    dst: DeviceRange<'a>,
) -> NormCall<'a> {
    NormCall {
        src: src.range(0, row).unwrap(),
        dst,
        health: role.gate_health.range(0, WORD).unwrap(),
    }
}

/// The separate launches of the chain of one role: the bits the fused chain must reproduce.
/// `input` is the role's input row, which another role may share.
fn separate(
    ctx: &CudaContext,
    stream: &DeviceStream,
    role: &Role,
    input: &DeviceBuffer,
    Shape { width, rank }: Shape,
) {
    let row = width * WORD;
    let norm = |src, dst| norm_row(role, row, src, dst);
    ctx.inkling_stream_norm_rows(
        stream,
        &[
            norm(input, role.gate.range(0, row).unwrap()),
            norm(&role.message, role.gate.range(row, row).unwrap()),
        ],
        (width, GATE_EPS),
    )
    .unwrap();
    ctx.dense_gemv_group(
        stream,
        &[
            role.gemv((&role.down, &role.gate, &role.reduced), [rank, 2 * width]),
            role.gemv((&role.gain_w, &role.reduced, &role.gain), [width, rank]),
            role.gemv((&role.forget_w, &role.reduced, &role.forget), [width, rank]),
        ],
    )
    .unwrap();
    ctx.inkling_controller_norm_rows(
        stream,
        std::slice::from_ref(&[
            input.range(0, row).unwrap(),
            role.message.range(0, row).unwrap(),
            role.gamma.range(0, row).unwrap(),
            role.gain.range(0, row).unwrap(),
            role.forget.range(0, row).unwrap(),
            role.out.range(0, row).unwrap(),
        ]),
        role.health.range(0, WORD).unwrap(),
        (width, EPS),
    )
    .unwrap();
}

/// `input` is the role's input row: its own, or the first role's when the bank shares one.
fn gate_role<'a>(role: &'a Role, input: &'a DeviceBuffer) -> GateRole<'a> {
    let word = WORD;
    GateRole {
        input: input.range(0, input.bytes()).unwrap(),
        message: role.message.range(0, role.message.bytes()).unwrap(),
        gate: role.gate.range(0, role.gate.bytes()).unwrap(),
        down: role.down.range(0, role.down.bytes()).unwrap(),
        reduced: role.reduced.range(0, role.reduced.bytes()).unwrap(),
        health: [
            role.gate_health.range(0, word).unwrap(),
            role.gate_health.range(0, word).unwrap(),
        ],
    }
}

/// The input row of `role` among `roles`: the first role's when the bank shares one.
fn input_of<'a>(roles: &'a [Role], role: &'a Role, shared: bool) -> &'a DeviceBuffer {
    if shared { &roles[0].input } else { &role.input }
}

fn all(buffer: &DeviceBuffer) -> DeviceRange<'_> {
    buffer.range(0, buffer.bytes()).unwrap()
}

fn apply_role<'a>(role: &'a Role, input: &'a DeviceBuffer) -> ApplyRole<'a> {
    ApplyRole {
        input: all(input),
        message: all(&role.message),
        gamma: all(&role.gamma),
        reduced: all(&role.reduced),
        gain_w: all(&role.gain_w),
        forget_w: all(&role.forget_w),
        gain: all(&role.gain),
        forget: all(&role.forget),
        out: all(&role.out),
        health: role.health.range(0, WORD).unwrap(),
    }
}

/// What the chain wrote: gate, reduced, gain, forget, out and the two health words.
fn written(stream: &DeviceStream, role: &Role) -> Vec<Vec<f32>> {
    [
        &role.gate,
        &role.reduced,
        &role.gain,
        &role.forget,
        &role.out,
        &role.health,
        &role.gate_health,
    ]
    .map(|buffer| {
        let mut out = vec![0u8; buffer.bytes()];
        buffer.read_at(stream, 0, &mut out).unwrap();
        floats(&out)
    })
    .to_vec()
}

fn clear(stream: &DeviceStream, roles: &[&Role]) {
    for role in roles {
        for buffer in [
            &role.gate,
            &role.reduced,
            &role.gain,
            &role.forget,
            &role.out,
            &role.health,
            &role.gate_health,
        ] {
            buffer
                .write_at(stream, 0, &vec![0u8; buffer.bytes()])
                .unwrap();
        }
    }
}

#[test]
fn fused_controller_chain_is_the_separate_launches_bit_for_bit() {
    if !testkit::device_present("fused_controller_chain_is_the_separate_launches_bit_for_bit") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x1d3c_c017_0a as u64);
    // (shape, roles, shared input): the two Mini shapes first (the MLP controller's one role,
    // the attention bank's three), then other ranks, which only the gate-down launch takes.
    for (shape, roles, shared) in [
        (
            Shape {
                width: 2304,
                rank: 64,
            },
            1,
            false,
        ),
        (
            Shape {
                width: 2304,
                rank: 64,
            },
            3,
            true,
        ),
        (
            Shape {
                width: 2304,
                rank: 64,
            },
            3,
            false,
        ),
        (
            Shape {
                width: 48,
                rank: 16,
            },
            2,
            false,
        ),
        (Shape { width: 64, rank: 8 }, 4, false),
    ] {
        let roles: Vec<Role> = (0..roles)
            .map(|_| Role::new(&ctx, &stream, &mut rng, shape, false))
            .collect();
        let tickets = zeros(&ctx, roles.len());
        for role in &roles {
            separate(&ctx, &stream, role, input_of(&roles, role, shared), shape);
        }
        let expected: Vec<_> = roles.iter().map(|role| written(&stream, role)).collect();
        clear(&stream, &roles.iter().collect::<Vec<_>>());
        let gates: Vec<_> = roles
            .iter()
            .map(|role| gate_role(role, input_of(&roles, role, shared)))
            .collect();
        let ticket = tickets.range(0, roles.len() * WORD).unwrap();
        ctx.decode_controller_gate_down(
            &stream,
            &gates,
            ticket,
            [shape.width, shape.rank],
            GATE_EPS,
        )
        .unwrap();
        if shape.rank == PAIR_RANK {
            let applies: Vec<_> = roles
                .iter()
                .map(|role| apply_role(role, input_of(&roles, role, shared)))
                .collect();
            ctx.decode_controller_pair_norm(&stream, &applies, ticket, shape.width, EPS)
                .unwrap();
        }
        // The pair launch's rows (gain, forget, out, health) exist only at its rank.
        let rows: &[usize] = if shape.rank == PAIR_RANK {
            &[0, 1, 2, 3, 4, 5, 6]
        } else {
            &[0, 1, 6]
        };
        for (role, expected) in roles.iter().zip(&expected) {
            let got = written(&stream, role);
            for &i in rows {
                assert_eq!(
                    got[i],
                    expected[i],
                    "row {i}, width {}, rank {}, {} roles",
                    shape.width,
                    shape.rank,
                    roles.len()
                );
            }
        }
        let mut back = vec![0u8; roles.len() * WORD];
        tickets.read_at(&stream, 0, &mut back).unwrap();
        assert!(
            back.iter().all(|&b| b == 0),
            "tickets are cleared for the next chain"
        );
    }
}

#[test]
fn fused_controller_chain_reports_a_non_finite_role_like_the_separate_launches() {
    if !testkit::device_present(
        "fused_controller_chain_reports_a_non_finite_role_like_the_separate_launches",
    ) {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0xbad_001e as u64);
    let shape = Shape {
        width: 2304,
        rank: 64,
    };
    let roles: Vec<Role> = [false, true]
        .into_iter()
        .map(|broken| Role::new(&ctx, &stream, &mut rng, shape, broken))
        .collect();
    for role in &roles {
        separate(&ctx, &stream, role, &role.input, shape);
    }
    let expected: Vec<_> = roles.iter().map(|role| written(&stream, role)).collect();
    clear(&stream, &roles.iter().collect::<Vec<_>>());
    let tickets = zeros(&ctx, roles.len());
    let gates: Vec<_> = roles
        .iter()
        .map(|role| gate_role(role, &role.input))
        .collect();
    ctx.decode_controller_gate_down(
        &stream,
        &gates,
        tickets.range(0, 8).unwrap(),
        [shape.width, shape.rank],
        GATE_EPS,
    )
    .unwrap();
    let applies: Vec<_> = roles
        .iter()
        .map(|role| apply_role(role, &role.input))
        .collect();
    ctx.decode_controller_pair_norm(
        &stream,
        &applies,
        tickets.range(0, 8).unwrap(),
        shape.width,
        EPS,
    )
    .unwrap();
    for (index, (role, expected)) in roles.iter().zip(&expected).enumerate() {
        let got = written(&stream, role);
        let quiet_nan = f32::from_bits(0x7fc0_0000);
        let same_bits =
            |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(same_bits(&got[5], &expected[5]), "health of role {index}");
        assert!(
            same_bits(&got[6], &expected[6]),
            "gate health of role {index}"
        );
        if index == 1 {
            assert_eq!(
                got[5][0].to_bits(),
                quiet_nan.to_bits(),
                "the broken role fails"
            );
            for (name, (got, expected)) in ["reduced", "gain", "forget", "out"]
                .iter()
                .zip([1, 3, 4, 5].map(|i| (&got[i], &expected[i])))
            {
                assert!(
                    got.iter()
                        .zip(expected.iter())
                        .all(|(g, e)| g.to_bits() == e.to_bits()),
                    "{name} of the broken role"
                );
            }
        }
    }
}

#[test]
fn fused_controller_chain_refuses_overlapping_rows_before_any_launch() {
    if !testkit::device_present("fused_controller_chain_refuses_overlapping_rows_before_any_launch")
    {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x0e41_a9 as u64);
    let shape = Shape {
        width: 64,
        rank: 64,
    };
    let role = Role::new(&ctx, &stream, &mut rng, shape, false);
    let tickets = zeros(&ctx, 2);
    let ticket = tickets.range(0, 4).unwrap();
    let gate = gate_role(&role, &role.input);
    let alias = ApplyRole {
        gain: role.forget.range(0, 64 * WORD).unwrap(),
        ..apply_role(&role, &role.input)
    };
    assert!(
        ctx.decode_controller_gate_down(&stream, &[gate], ticket, [64, 64], GATE_EPS)
            .is_ok()
    );
    let error = ctx
        .decode_controller_pair_norm(&stream, &[alias], ticket, 64, EPS)
        .unwrap_err();
    assert!(
        matches!(error, EngError::BadRequest(ref message) if message.contains("overlap")),
        "{error:?}"
    );
    let error = ctx
        .decode_controller_gate_down(
            &stream,
            &[gate],
            tickets.range(0, 8).unwrap(),
            [64, 64],
            GATE_EPS,
        )
        .unwrap_err();
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
    let error = ctx
        .decode_controller_gate_down(&stream, &[gate], ticket, [64, 63], GATE_EPS)
        .unwrap_err();
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
    let mut out = vec![0u8; 64 * WORD];
    role.out.read_at(&stream, 0, &mut out).unwrap();
    assert!(
        floats(&out).iter().all(|v| v.to_bits() == 0),
        "a refused call changes nothing"
    );
}
