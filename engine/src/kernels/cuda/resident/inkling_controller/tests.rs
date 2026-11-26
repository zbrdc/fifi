mod host;

use super::Arithmetic;
use crate::kernels::cuda::{CudaContext, device_present, lane_norm};

const EPS: f32 = 1e-5;

fn pack(roles: usize, width: usize, broken: Option<usize>) -> Vec<f32> {
    let mut words = vec![0.0; roles * width * 6];
    for role in 0..roles {
        for i in 0..width {
            let value = |seed| (((role * 13 + i * 7 + seed) % 97) as f32 - 48.0) / 32.0;
            for (j, seed) in [1, 3, 7].into_iter().enumerate() {
                words[(role * 3 + j) * width + i] = value(seed);
            }
            for (j, seed) in [11, 17].into_iter().enumerate() {
                words[(3 * roles + role * 2 + j) * width + i] = value(seed);
            }
        }
    }
    if let Some(role) = broken {
        words[(3 * roles + role * 2) * width] = f32::INFINITY;
    }
    words
}

fn run(
    ctx: &CudaContext,
    shape: [usize; 2],
    input: &[f32],
    eps: f32,
    arithmetic: Arithmetic,
) -> (Vec<f32>, Vec<f32>) {
    let [width, roles] = shape;
    let stream = ctx.stream().unwrap();
    let words = input.len();
    let workspace = ctx.allocate(words * 4).unwrap();
    let health = ctx.allocate(roles * 4).unwrap();
    let bytes: Vec<_> = input.iter().flat_map(|v| v.to_ne_bytes()).collect();
    workspace.write_at(&stream, 0, &bytes).unwrap();
    ctx.inkling_controller_norm_with_arithmetic(
        &stream,
        workspace.range(0, words * 4).unwrap(),
        health.range(0, roles * 4).unwrap(),
        (shape, eps),
        arithmetic,
    )
    .unwrap();
    let mut output = vec![0u8; roles * width * 4];
    workspace
        .read_at(&stream, 5 * roles * width * 4, &mut output)
        .unwrap();
    let mut status = vec![0u8; roles * 4];
    health.read_at(&stream, 0, &mut status).unwrap();
    let floats = |bytes: &[u8]| {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect()
    };
    (floats(&output), floats(&status))
}

#[test]
fn batched_controller_norm_matches_independent_roles_and_reports_health() {
    if !device_present("batched controller norm") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let width = 37;
    for roles in [1, 3, 6, 195, 1536] {
        let packed = pack(roles, width, None);
        let (out, health) = run(&ctx, [width, roles], &packed, EPS, Arithmetic::Generic);
        assert!(health.iter().all(|h| h.to_bits() == 0));
        for role in [0, roles / 2, roles - 1] {
            let mut single = vec![0.0; width * 6];
            single[..width * 3].copy_from_slice(&packed[role * 3 * width..(role + 1) * 3 * width]);
            single[width * 3..width * 5].copy_from_slice(
                &packed[(3 * roles + 2 * role) * width..(3 * roles + 2 * role + 2) * width],
            );
            let (expected, state) = run(&ctx, [width, 1], &single, EPS, Arithmetic::Generic);
            assert_eq!(state[0].to_bits(), 0);
            for (a, b) in out[role * width..(role + 1) * width].iter().zip(expected) {
                assert_eq!(a.to_bits(), b.to_bits(), "roles={roles} role={role}");
            }
        }
    }
    let (_, health) = run(
        &ctx,
        [width, 6],
        &pack(6, width, Some(2)),
        EPS,
        Arithmetic::Generic,
    );
    assert!(health[2].is_nan());
    assert!(
        health
            .iter()
            .enumerate()
            .all(|(i, h)| i == 2 || h.to_bits() == 0)
    );
}

#[test]
fn cpu_parity_controller_norm_matches_host_on_exact_taps() {
    if !device_present("Inkling CPU parity controller norm") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    for (width, eps) in [(1, EPS), (513, EPS), (4096, EPS), (513, f32::from_bits(1))] {
        let shape = [width, host::ROLES];
        let packed = host::exact_tap_fixture(width);
        let (out, health) = run(&ctx, shape, &packed, eps, Arithmetic::CpuParity);
        assert!(health.iter().all(|h| h.to_bits() == 0));
        let expected = host::controller_norm(&packed, width, eps);
        for (index, (device, reference)) in out.into_iter().zip(expected).enumerate() {
            assert_eq!(
                device.to_bits(),
                reference.to_bits(),
                "width={width}, eps={eps:e}, role={}, col={}",
                index / width,
                index % width,
            );
        }
    }
}

#[test]
fn cpu_parity_controller_norm_reports_invalid_roles() {
    if !device_present("Inkling CPU parity controller health") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let (width, roles) = (257usize, 10usize);
    let mut words = vec![0.0; roles * width * 6];
    let last = width - 1;
    let raw = |role: usize, field: usize| (role * 3 + field) * width;
    let coeff = |role: usize, field: usize| (3 * roles + role * 2 + field) * width;
    for role in 0..roles {
        words[raw(role, 2)..raw(role, 2) + width].fill(1.0);
    }
    words[coeff(1, 0) + last] = f32::INFINITY; // tanh masks bad gain
    words[coeff(2, 1) + last] = f32::NEG_INFINITY; // tanh masks bad forget
    words[coeff(3, 1) + last] = f32::NAN;
    words[raw(4, 0) + last] = f32::INFINITY;
    words[raw(5, 1) + last] = f32::NAN;
    words[raw(6, 2) + last] = f32::INFINITY;
    words[raw(7, 0) + last] = f32::MAX; // square overflows despite finite x
    words[raw(8, 0)] = 1.4e19;
    words[raw(8, 0) + last] = 1.4e19; // individually finite squares, overflowing sum
    words[raw(9, 0) + last] = 1.0;
    words[raw(9, 2) + last] = f32::MAX; // normalized value times gamma overflows

    let (out, health) = run(&ctx, [width, roles], &words, EPS, Arithmetic::CpuParity);
    assert_eq!(health[0].to_bits(), 0);
    for (role, h) in health.into_iter().enumerate().skip(1) {
        assert!(h.is_nan(), "expected failed health at role {role}");
    }
    for role in [1, 2, 7, 8] {
        assert!(
            out[role * width..(role + 1) * width]
                .iter()
                .all(|x| x.is_finite())
        );
    }
    // A finite sum and a finite epsilon can overflow when added.
    let packed = [1.4e19, 0.0, 1.0, 0.0, 0.0, 0.0];
    let (out, health) = run(&ctx, [1, 1], &packed, f32::MAX, Arithmetic::CpuParity);
    assert!(out[0].is_finite() && health[0].is_nan());
}

fn all(buffer: &crate::kernels::cuda::DeviceBuffer) -> crate::kernels::cuda::DeviceRange<'_> {
    buffer.range(0, buffer.bytes()).unwrap()
}

#[test]
fn prepare_controller_gates_are_the_lane_order_rms_norm_bits() {
    if !device_present("Inkling controller gate preparation") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let upload = |values: &[f32]| {
        let buffer = ctx.allocate(values.len() * 4).unwrap();
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        buffer.write_at(&stream, 0, &bytes).unwrap();
        buffer
    };
    let download = |buffer: &crate::kernels::cuda::DeviceBuffer| -> Vec<f32> {
        let mut bytes = vec![0u8; buffer.bytes()];
        buffer.read_at(&stream, 0, &mut bytes).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect()
    };
    for (width, roles) in [(1, 1), (257, 3), (2304, 4), (4096, 2)] {
        let mut pairs: Vec<f32> = (0..2 * roles * width)
            .map(|i| (((i * 7919 + 13) % 1009) as f32 - 504.0) / 97.0)
            .collect();
        // The last half's squares overflow while every value stays finite.
        let broken = 2 * roles - 1;
        if roles > 1 {
            pairs[broken * width] = 1.4e19;
            pairs[broken * width + width - 1] = 1.4e19;
        }
        let gamma: Vec<f32> = (0..width).map(|i| 1.0 + i as f32 / 4096.0).collect();
        let [pairs_d, gamma_d] = [upload(&pairs), upload(&gamma)];
        let triples = ctx.allocate(3 * roles * width * 4).unwrap();
        let gates = ctx.allocate(2 * roles * width * 4).unwrap();
        let health = ctx.allocate(2 * roles * 4).unwrap();
        ctx.inkling_prepare_controller(
            &stream,
            [
                all(&pairs_d),
                all(&gamma_d),
                all(&triples),
                all(&gates),
                all(&health),
            ],
            ([width, roles], EPS),
        )
        .unwrap();
        stream.synchronize().unwrap();
        let [gates, triples, health] = [download(&gates), download(&triples), download(&health)];
        for half in 0..2 * roles {
            let source = &pairs[half * width..][..width];
            let failed = roles > 1 && half == broken;
            assert_eq!(health[half].is_nan(), failed, "width={width} half={half}");
            if !failed {
                assert_eq!(health[half].to_bits(), 0, "width={width} half={half}");
                let expected = lane_norm(source, None, EPS);
                for (i, (device, host)) in gates[half * width..][..width]
                    .iter()
                    .zip(expected)
                    .enumerate()
                {
                    assert_eq!(
                        device.to_bits(),
                        host.to_bits(),
                        "width={width} half={half} col={i}"
                    );
                }
            }
            let (role, side) = (half / 2, half % 2);
            let raw = &triples[(3 * role + side) * width..][..width];
            assert!(
                raw.iter()
                    .zip(source)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            let raw_gamma = &triples[(3 * role + 2) * width..][..width];
            assert!(
                raw_gamma
                    .iter()
                    .zip(&gamma)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }
}

/// The rows entry (the caller's `[input, message, gamma, gain, forget, out]` per role) is the
/// packed launch bit for bit, with the same health; an output row that shares a byte with a row
/// of another role is refused before any launch.
#[test]
fn rows_controller_norm_equals_the_packed_launch_bit_for_bit() {
    use crate::EngError;

    if !device_present("Inkling controller norm rows") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let width = 2304;
    for roles in [1usize, 3] {
        let packed = pack(roles, width, None);
        let (expected, expected_health) =
            run(&ctx, [width, roles], &packed, EPS, Arithmetic::Generic);
        let words = packed.len();
        let workspace = ctx.allocate(words * 4).unwrap();
        let outs = ctx.allocate(roles * width * 4).unwrap();
        let health = ctx.allocate(roles * 4).unwrap();
        let bytes: Vec<_> = packed.iter().flat_map(|v| v.to_ne_bytes()).collect();
        workspace.write_at(&stream, 0, &bytes).unwrap();
        health.write_at(&stream, 0, &vec![0u8; roles * 4]).unwrap();
        fn row<'a>(
            buffer: &'a crate::kernels::cuda::DeviceBuffer,
            at: usize,
            width: usize,
        ) -> crate::kernels::cuda::DeviceRange<'a> {
            buffer.range(at * width * 4, width * 4).unwrap()
        }
        let rows: Vec<_> = (0..roles)
            .map(|role| {
                [
                    row(&workspace, role * 3, width),
                    row(&workspace, role * 3 + 1, width),
                    row(&workspace, role * 3 + 2, width),
                    row(&workspace, 3 * roles + role * 2, width),
                    row(&workspace, 3 * roles + role * 2 + 1, width),
                    row(&outs, role, width),
                ]
            })
            .collect();
        let mut clash = rows.clone();
        if roles > 1 {
            clash[1][5] = row(&workspace, 0, width);
            let error = ctx.inkling_controller_norm_rows(
                &stream,
                &clash,
                health.range(0, roles * 4).unwrap(),
                (width, EPS),
            );
            assert!(matches!(error, Err(EngError::BadRequest(_))), "{error:?}");
        }
        ctx.inkling_controller_norm_rows(
            &stream,
            &rows,
            health.range(0, roles * 4).unwrap(),
            (width, EPS),
        )
        .unwrap();
        let mut output = vec![0u8; roles * width * 4];
        outs.read_at(&stream, 0, &mut output).unwrap();
        let mut status = vec![0u8; roles * 4];
        health.read_at(&stream, 0, &mut status).unwrap();
        let floats = |bytes: &[u8]| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_ne_bytes(*b))
                .collect::<Vec<_>>()
        };
        let (output, status) = (floats(&output), floats(&status));
        for (a, b) in output.iter().zip(&expected) {
            assert_eq!(a.to_bits(), b.to_bits(), "roles {roles}");
        }
        assert_eq!(status, expected_health);
        let _ = stream;
    }
}
